import numpy as np
import matplotlib.pyplot as plt
'''
We show the establishment process of K-Nearnest Neighbor
'''

def creatDataset():
    group = np.array([[1, 1.1], [1, 1.0], [0, 0], [0, 0.1]])
    labels = ['A','A ','B','B ']
    return group, labels


if __name__ == "__main__":
    group, labels = creatDataset()
    x,y = [],[]
    for temp in group:
        x.append(temp[0])
        y.append(temp[1])

    plt.scatter(x,y)
    for temp in range(len(x)):
        plt.text(x[temp]-0.04, y[temp]-0.04, labels[temp])
    plt.show()