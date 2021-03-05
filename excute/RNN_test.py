from functions import data_preparation
from model import chemp_model, useRNN
from functions.test_functions import gray_ber
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


# --------------------------------------------- Dataset ------------------------------------------------------
class DetDataset(Dataset):
    def __init__(self, z, j_matrix, data_real, data_imag, transform=None):
        self.z = z
        self.j_matrix = j_matrix
        self.label = torch.cat([torch.from_numpy(data_real.T),
                                torch.from_numpy(data_imag.T)],
                               dim=1).long()
        self.transform = transform

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'z': torch.from_numpy(self.z[idx, :]).to(torch.float32),
            'j_matrix': torch.from_numpy(self.j_matrix[idx, :, :]).to(torch.float32),
            'label': self.label[idx, :]
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    TX = 8
    RX = 32
    N_TRAIN = 30000
    N_TEST = 2000
    TRAIN_SPLIT = 0.9
    RATE = 2
    ITERATIONS = 10
    EBN0_TRAIN = 10
    LENGTH = 2 ** RATE
    BATCH_SIZE = 5
    EPOCHS = 100
    HIDDEN_SIZE = 6 * TX
    RNN_LAYERS = 1

    # Here the data has already been divided by rx
    train_z, train_J, train_var, train_Data_real, train_Data_imag = data_preparation.get_data(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    test_z, test_J, test_var, test_Data_real, test_Data_imag = data_preparation.get_data(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_z, train_J, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_z, test_J, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------------------- Establish Network ----------------------------------------------
    chemp_model = useRNN.CHEMPModel(LENGTH, 2*TX, ITERATIONS, HIDDEN_SIZE)
    loss_fn = nn.NLLLoss()
    optim_chemp = torch.optim.Adam(chemp_model.parameters(), lr=0.00001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim_chemp, step_size=10, gamma=0.4)

    p_ini = torch.ones([BATCH_SIZE, LENGTH, 2*TX]) / LENGTH  # initial probability
    h_ini = torch.zeros([RNN_LAYERS, BATCH_SIZE, HIDDEN_SIZE])

    # ------------------------------------- Train ----------------------------------------------------------
    print('Begin Training:')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = (data['z'], data['j_matrix']), data['label']

            # zero the parameter gradients
            optim_chemp.zero_grad()

            # forward + backward + optimize
            prob, h = chemp_model(inputs, p_ini, h_ini, train_var)
            loss = loss_fn(torch.log(prob), labels)
            loss.backward()
            optim_chemp.step()

            # print statistics
            running_loss += loss.item()
            if i % (round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)) == round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE) - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = (data['z'], data['j_matrix']), data['label']
                prob, h = chemp_model(inputs, p_ini, h_ini, train_var)
                loss = loss_fn(torch.log(prob), labels)
                val_loss += loss.numpy()
                val_steps += 1
        print('validation loss: %.3f' % (val_loss / val_steps))
        scheduler.step()

    print('Training finished')
    # --------------------------------------------- Test ---------------------------------------------------------------
    with torch.no_grad():
        chemp_model.eval()
        test_loss = 0.0
        test_steps = 0
        predictions = []
        for i, data in enumerate(test_loader, 0):
            inputs, labels = (data['z'], data['j_matrix']), data['label']
            prob, h = chemp_model(inputs, p_ini, h_ini, test_var)
            loss = loss_fn(torch.log(prob), labels)
            _, prediction = torch.max(prob, dim=1)
            predictions += [prediction]
            test_loss += loss.numpy()
            test_steps += 1
        print('test loss: %.3f' % (test_loss / test_steps))

        predictions = torch.cat(predictions).numpy()

        ber = gray_ber(predictions, test_Data_real, test_Data_imag, rate=RATE)
        print(ber)
    # ------------------------------------------------- Whole Test -----------------------------------------------------
    chemp_model.eval()
    TEST_EBN0 = np.linspace(0, 15, 16)
    BER = []
    for ebn0 in TEST_EBN0:
        test_z, test_J, test_var, test_Data_real, test_Data_imag = data_preparation.get_data(tx=TX, rx=RX, K=N_TEST,
                                                                                             rate=RATE,
                                                                                             EbN0=ebn0)
        test_set = DetDataset(test_z, test_J, test_Data_real, test_Data_imag)
        test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            test_loss = 0.0
            test_steps = 0
            predictions = []
            for i, data in enumerate(test_loader, 0):
                inputs, labels = (data['z'], data['j_matrix']), data['label']
                prob, h = chemp_model(inputs, p_ini, h_ini, test_var)
                loss = loss_fn(torch.log(prob), labels)
                _, prediction = torch.max(prob, dim=1)
                predictions += [prediction]
                test_loss += loss.numpy()
                test_steps += 1
            print('test loss: %.3f' % (test_loss / test_steps))

            predictions = torch.cat(predictions).numpy()
            ber = gray_ber(predictions, test_Data_real, test_Data_imag, rate=RATE)
            BER += [ber]
    # ——----------------------------------- Save Model & Data ----------------------------------------------------------
    PATH = '../pretrained_model/GRU_CHEMP/rx%i/tx%i/EbN0_Train%i/iterations%i/batch_size%i/hidden_size%i' % (RX, TX,
                                                                                                                EBN0_TRAIN,
                                                                                                                ITERATIONS,
                                                                                                                BATCH_SIZE,
                                                                                                                HIDDEN_SIZE)
    os.makedirs(PATH)
    data_ber = pd.DataFrame(BER, columns=['BER'])
    data_ber.to_csv(PATH + str('/ber2.csv'))
    torch.save(chemp_model.state_dict(), PATH+str('/model2.pt'))
    # use the following line to load model
    chemp_model.load_state_dict(torch.load(PATH + str('/model1.pt')))
