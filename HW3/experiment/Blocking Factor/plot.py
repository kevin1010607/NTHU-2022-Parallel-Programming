import matplotlib.pyplot as plt
import json
import os

if not os.path.exists("image/"): 
    os.mkdir("image/")

def plot_bar(data, factor, N, x_label, y_label, title):
    x = range(N)
    plt.figure()
    plt.bar(x, data, 0.4)
    plt.xticks(x, factor)
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
factor = [i['factor'] for i in data]
gops = [i['inst_integer']/(i['time']*1e-3)/1e9 for i in data]
shared_load_throughput = [i['shared_load_throughput'] for i in data]
shared_store_throughput = [i['shared_store_throughput'] for i in data]
gld_throughput = [i['gld_throughput'] for i in data]
gst_throughput = [i['gst_throughput'] for i in data]

x_label = "Blocking Factor"

plot_bar(gops, factor, N, x_label, "Integer GOPS", "Integer GOPS")
plot_bar(shared_load_throughput, factor, N, x_label, "Throughput (GB/s)", "Shared Load Throughput")
plot_bar(shared_store_throughput, factor, N, x_label, "Throughput (GB/s)", "Shared Store Throughput")
plot_bar(gld_throughput, factor, N, x_label, "Throughput (GB/s)", "Global Load Throughput")
plot_bar(gst_throughput, factor, N, x_label, "Throughput (GB/s)", "Global Store Throughput")