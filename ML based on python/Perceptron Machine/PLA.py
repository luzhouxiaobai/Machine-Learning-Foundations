import numpy as np

'''
Perceptron Learning Algoritm
'''

def data_process(file):
    data = np.loadtxt(file)
    train_data = [temp[1:] for temp in data]
    train_label = [temp[0] for temp in data]
    return train_data, train_label

#原始形态的PLA
def pla(train_data, train_label):
    temp = len(train_data[0])

    # initial 0-n
    w = np.array([0 for i in range(temp+1)])
    temp = len(train_data)
    while index < temp:
        tt = train_data[index].tolist()
        tt.append(1.0)
        tindex = np.array(tt)
        if np.dot(w, tindex) * train_label[index] > 0:
            index = index + 1
            continue
        else:
            w = w + train_label[index] * tindex
            index = index + 1
        if index + 1 == temp:
            index = 0

    return w

#对偶形式的PLA
def dpla(train_data,train_label):
    l,l1 = len(train_data),len(train_data[0])
    trans = []
    for i in range(l):
        temp = []
        for j in range(l):
            tr = np.dot(train_data[i],train_data[j])
            temp.append(tr)
        trans.append(temp)
    a = [0 for i in range(l)]
    b = 0
    k = 1
    while i < l:
        temp = [0 for m in range(l1)]
        for j in range(l):
            x = [a[j] * train_label[j] * x for x in train_data[j]]
            temp1 = [temp[i]+x[i] for i in range(l1)]

        res = np.dot(np.array(temp1),train_data[i]) + b
        if res * train_label[i] <= 0:
            a = [x+k for x in a]
            b = b + train_label[i]

            if i + 1 == l:
                i = 0
            else:
                i = i + 1
        else:
            i = i + 1
            continue

    return a, b





if __name__ == "__main__":
    file = "data.txt"
    s1, s2 = data_process(file)
    w,b = dpla(s1, s2)
    print(w,b)