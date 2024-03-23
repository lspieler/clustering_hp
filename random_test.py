from matplotlib import pyplot as plt
import random


while True:
    prices = [510]
    sd = 0.1

    for i in range(390):
        prices.append(prices[-1] + random.gauss(0, sd))

    # prices = [prices[i] for i in range(0, len(prices), 390)]
    plt.plot(prices)
    plt.show()

    # make classifier to learn whether the data is random or not

