from functions import data_preparation
from model import RNN_CHEMP
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.utils.data import Dataset

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
    N_TRAIN = 40000
    N_TEST = 1000
    TRAIN_SPLIT = 0.9
    RATE = 2
    N_LAYERS = 5
    LSTM_LAYERS = 3
    EBN0_TRAIN = 10
    LENGTH = 2 ** RATE
    BATCH_SIZE = 20
    EPOCHS = 200

    # Here the data has already been divided by 2tx
    train_z, train_J, train_var, train_Data_real, train_Data_imag = data_preparation.get_data(tx=TX, rx=RX, K=N_TRAIN, rate=RATE, EbN0=EBN0_TRAIN)
    test_z, test_J, test_var, test_Data_real, test_Data_imag = data_preparation.get_data(tx=TX, rx=RX, K=N_TEST, rate=RATE, EbN0=EBN0_TRAIN)

    train_set = DetDataset(train_z, train_J, train_Data_real, train_Data_imag)
    test_set = DetDataset(test_z, test_J, test_Data_real, test_Data_imag)

    trainSet, valSet = Data.random_split(train_set, [int(N_TRAIN * TRAIN_SPLIT), round(N_TRAIN * (1 - TRAIN_SPLIT))])
    train_loader = Data.DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = Data.DataLoader(valSet, batch_size=BATCH_SIZE, shuffle=False)

    writer = SummaryWriter('../log/chemp/test1')

    # ------------------------------------- Establish Network ----------------------------------------------
    chemp = RNN_CHEMP.RecurrentCHEMP(N_LAYERS, TX, LENGTH, train_var, LSTM_LAYERS)
    loss_fn = nn.NLLLoss()
    optim_chemp = torch.optim.Adam(chemp.parameters(), lr=0.00001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim_chemp, step_size=50, gamma=0.1)
    # # Froze the delta to bound the prob under 1
    # for layer in range(N_LAYERS):
    #     getattr(chemp_model, 'chemp_layer_'+str(layer)).delta.requires_grad = False

    # ------------------------------------- Train ----------------------------------------------------------
    print('Begin Training:')
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = (data['z'], data['j_matrix']), data['label']

            # zero the parameter gradients
            # optim_chemp.zero_grad()

            # forward + backward + optimize

            prob = chemp(inputs)
            loss = loss_fn(torch.log(prob), labels)
            loss.backward()
            optim_chemp.step()

            # print statistics
            running_loss += loss.item()
            if i % (round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)) == round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE) - 1:
                writer.add_scalar('loss/train_loss', running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE), epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / round(N_TRAIN * TRAIN_SPLIT / BATCH_SIZE)))
                running_loss = 0.0
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = (data['z'], data['j_matrix']), data['label']
                prob = chemp(inputs)
                loss = loss_fn(torch.log(prob), labels)
                val_loss += loss.numpy()
                val_steps += 1
        writer.add_scalar('loss/val_loss', val_loss/val_steps, epoch)
        print('validation loss: %.3f' % (val_loss / val_steps))
        scheduler.step()

    print('Training finished')
