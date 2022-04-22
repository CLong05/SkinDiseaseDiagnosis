import pandas as pd
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('logs/2022-04-21_13:24/train_data_99.csv', index_col=0)
    loss = data.iloc[:, 0]
    train_accuracy = data.iloc[:, 1]
    plt.plot(loss, label='loss')
    plt.plot(train_accuracy, label='train_accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
