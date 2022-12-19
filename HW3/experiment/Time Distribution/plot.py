import matplotlib.pyplot as plt
import json
import os

if not os.path.exists("image/"): 
    os.mkdir("image/")

def plot(data, color, label, n, N, x_label, y_label, title):
    plt.figure(figsize=(8, 7))
    for i in range(len(data)):
        plt.plot(n, data[i], color=color[i], label=label[i], marker='.')
    plt.legend(loc='upper left')
    plt.xticks(n)
    plt.ylim(bottom=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for a, b in zip(n, data[0]):
        plt.text(a, b+0.05, f'{b:.2f}', ha='center', va= 'bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'./image/{title}.png', dpi=300)
    plt.show()

with open('data.json') as f:
    data = json.load(f)

N = len(data)
n = [i['n'] for i in data]
total_time = [i['total_time'] for i in data]
input_time = [i['input_time'] for i in data]
output_time = [i['output_time'] for i in data]
h2d = [i['h2d'] for i in data]
d2h = [i['d2h'] for i in data]
compute = [i['compute'] for i in data]

data = [total_time, input_time, output_time, h2d, d2h, compute]
color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
label = ['Total', 'Input', 'Output', 'H2D', 'D2H', 'Compute']

plot(data, color, label, n, N, "n", "Time (s)", "Time Distribution")