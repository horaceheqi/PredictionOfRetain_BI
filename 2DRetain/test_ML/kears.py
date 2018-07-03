import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

if __name__ == '__main__':

    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    print(train_x.shape, train_y.shape)
    print(test_x.shape, test_y.shape)