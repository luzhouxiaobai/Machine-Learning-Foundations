import numpy as np

'''
Pocket Algorithm
'''

def data_process(file):
    data = np.loadtxt(file)
    train_data = []
    train_label = []
    data = data.tolist()
    temp = len(data)
    for i in range(temp):
        train_data.append(data[i][1:])
        train_label.append(data[i][0])
        train_data[i].append(1)
    return np.array(train_data), np.array(train_label)

def AP(train_data, train_label):
    temp = len(train_data[0])
    w = np.array([0 for i in range(temp)])
    temp = len(train_data)

    for index in range(temp):
        tindex = train_data[index]
        if np.dot(w, tindex) * train_label[index] > 0:
            continue
        else:
            w_temp = w + train_label[index] * tindex
            count1, count2 = 0, 0
            for i in range(temp):
                if np.dot(w, train_data[i]) * train_label[index] <= 0:
                    count1 = count1 + 1
                if np.dot(w_temp, train_data[i]) * train_label[index] <= 0:
                    count2 = count1 + 2
            if count1 > count2: w = w_temp

    return w

if __name__ == "__main__":
    file = "data.txt"
    s1, s2 = data_process(file)
    w = AP(s1, s2)
    print(w)