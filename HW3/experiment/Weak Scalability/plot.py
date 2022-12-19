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
    plt.xlim(left=-0.5, right=1.5)
    plt.ylim(bottom=0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    for a, b in zip(x, data):
        plt.text(a, b+0.05, f'{b:.2f}', ha='center', va= 'bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(f'./image/{title}.png', dpi=300)
    plt.show()

def plot_speedup(data, X, N, x_label, y_label, title):
    x = range(N)
    plt.figure()
    plt.plot(x, data, marker='.')
    plt.xticks(x, X)
    plt.xlim(left=-0.5, right=1.5)
    plt.ylim(bottom=0, top=2)
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
gpu = [i['gpu'] for i in data]
time = [i['time'] for i in data]
base = time[0]
speedup = [base/i*idx for idx, i in enumerate(time, 1)]

x_label = "GPU"

plot_bar(time, gpu, N, x_label, "Time (s)", "Weak Scalability - Time")
plot_speedup(speedup, gpu, N, x_label, "Speed up (factor)", "Weak Scalability - Speed Up")