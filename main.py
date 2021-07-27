import NeuralNetwork.Network
import numpy as np
import matplotlib.pyplot as plt

def main():
    lol = NeuralNetwork.Network.NeuralNetwork(2, 1, 5, 0.3)
    col = 50000
    forplot = np.zeros(col)
    for i in range(col):
        inp = np.random.randint(0, 2, (2, 1))
        res = int(inp[0] != inp[1])
        forplot[i] = lol.train(inp, [res])
    fig = plt.figure()
    plt.plot(forplot)
    plt.show()
if __name__ == "__main__":
    main()
