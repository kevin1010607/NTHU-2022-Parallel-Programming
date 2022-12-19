import matplotlib.pyplot as plt
import json
import os

if not os.path.exists("image/"): 
    os.mkdir("image/")

def plot_bar(data, X, N, x_label, y_label, title):
    x = range(N)
    plt.figure(figsize=(8, 7))
    plt.bar(x, data, 0.4)
    plt.xticks(x, X, rotation=45)
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
    plt.figure(figsize=(8, 7))
    plt.plot(x, data, marker='.')
    plt.xticks(x, X, rotation=45)
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
optimization = [i['optimization'] for i in data]
time = [i['time'] for i in data]
base = time[0]
speedup = [base/i for i in time]

x_label = "Optimization"

plot_bar(time, optimization, N, x_label, "Time (s)", "Performance Optimization")
plot_speedup(speedup, optimization, N, x_label, "Speed up (factor)", "Speed Up")