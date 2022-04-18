#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

columns = ['Farrah', 'Fred', 'Felicia']
fruit_label = ['apples', 'bananas', 'oranges', 'peaches']

plt.bar(columns, fruit[0], color="red", width=0.5)
plt.bar(columns, fruit[1], bottom=fruit[0], color="yellow", width=0.5)
plt.bar(columns, fruit[2], bottom=fruit[0] +
        fruit[1], color="#ff8000", width=0.5)
plt.bar(columns, fruit[2], bottom=fruit[0] +
        fruit[1]+fruit[2], color="#ffe5b4", width=0.5)
plt.ylim(0, 80)
plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.legend(fruit_label)
plt.show()
