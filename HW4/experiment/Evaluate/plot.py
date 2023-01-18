import matplotlib.pyplot as plt
import json
import os

if not os.path.exists("image/"): 
    os.mkdir("image/")

def plot_bar(data, X, N, x_label, y_label, title):
    x = range(N)
    plt.figure()
    plt.bar(x, data, 0.4)
    plt.xticks(x, X)
    plt.ylim(bottom=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for a, b in zip(x, data):
        plt.text(a, b+0.05, f'{b:.2f}', ha='center', va= 'bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'./image/{title}.png', dpi=300)
    plt.show()

with open('data.json') as f:
    data = json.load(f)

N = len(data)
testcase = [i['testcase'] for i in data]
time = [i['time'] for i in data]

x_label = "Testcase"

plot_bar(time, testcase, N, x_label, "Time (s)", "Testcase - Time")
