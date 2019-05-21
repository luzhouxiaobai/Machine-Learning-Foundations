import numpy as np

'''
Perceptron Learning Algoritm
'''

def data_process(file):
    data = np.loadtxt(file)
    train_data = [temp[1:] for temp in data]
    train_label = [temp[0] for temp in data]
    return train_data, train_label

def pla(train_data, train_label):
    temp = len(train_data[0])

    # initial 0-n
    w = np.array([0 for i in range(temp+1)])
    temp = len(train_data)
    for index in range(temp):
        tt = train_data[index].tolist()
        tt.append(1.0)
        tindex = np.array(tt)
        if np.dot(w, tindex) * train_label[index] > 0:
            continue
        else:
            w = w + train_label[index] * tindex
        if index + 1 == temp:
            index = 0

    return w

if __name__ == "__main__":
    file = "data.txt"
    s1, s2 = data_process(file)
    w = pla(s1, s2)
    print(w)