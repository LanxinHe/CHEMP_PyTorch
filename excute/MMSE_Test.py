from functions.data_preparation import get_y_h
import numpy as np
from functions.test_functions import gray_ber

TX = 16
RX = 16
RATE = 1
LENGTH = 2 ** RATE
K = 2000
EbN0 = 10
y, h, data_real, data_imag, var = get_y_h(rx=RX, tx=TX, K=K, rate=RATE, EbN0=EbN0)
predictions = []
for k in range(K):
    H = h[k]
    receive = y[k]
    G = np.matmul(np.linalg.pinv(np.matmul(H.T, H) + (var * 3 / 2 / (LENGTH ** 2 - 1)*np.eye(2*TX))),
                  H.T)
    x_hat = np.dot(G, receive)
    predictions += [x_hat]
predictions = np.array(predictions)
predictions = np.round((predictions + LENGTH - 1)/2)
ber = gray_ber(predictions, data_real, data_imag, RATE)
print(ber)
