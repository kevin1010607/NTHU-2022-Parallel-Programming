import matplotlib.pyplot as plt

def plot_speed_up(time, N, base, x_label, title):
    x = range(1, N+1)
    speedup = list(map(lambda x: base/x, time))
    plt.figure()
    plt.plot(x, speedup, color='blue', marker='.')
    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.xlabel(x_label)
    plt.ylabel('Speed up (factor)')
    plt.title(f'Scalability ({title})')
    plt.savefig(f'./image/speed_up_{title}.png', dpi=300)
    plt.show()

def plot_load_balancing(time, N, x_label, title):
    x = range(1, N+1)
    plt.figure()
    plt.plot(x, time, color='blue', marker='.')
    plt.xticks(x)
    plt.ylim(bottom=0, top=max(time)+4)
    plt.xlabel(x_label)
    plt.ylabel('Time (sec)')
    plt.title(f'Load Balancing ({title})')
    plt.savefig(f'./image/load_balancing_{title}.png', dpi=300)
    plt.show()

def plot_load_balancing_multi(time, N, x_label, title):
    x = range(1, N+1)
    label = [f'{i+1} node' for i in range(4)]
    color = ['blue', 'red', 'green', 'purple']
    plt.figure()
    for i in range(4):
        plt.plot(x, time[i], color=color[i], label=label[i], marker='.')
    plt.legend(loc='upper left')
    plt.xticks(x)
    plt.ylim(bottom=0, top=max(time[0])+4)
    plt.xlabel(x_label)
    plt.ylabel('Time (sec)')
    plt.title(f'Load Balancing ({title})')
    plt.savefig(f'./image/load_balancing_{title}.png', dpi=300)
    plt.show()

# Single node for pthread (1~12 thread)
data = [[0]*(i+2) for i in range(12)]

for i in range(12):
    for j in range(i+2):
        data[i][j] = float(input())

total_time = [i[-1] for i in data]
thread_time = data[-1][:-1]
base = total_time[0]
x_label = 'Number of thread'
x_label_load = 'Id of thread'
title = 'single-node, pthread'

plot_speed_up(total_time, 12, base, x_label, title)
plot_load_balancing(thread_time, 12, x_label_load, title)

# Multi node for hybrid (1~4 node, 1 process per node, 12 thread per process)
data = [[0]*(4+12) for i in range(4)]

for i in range(4):
    idx = 0
    for j in range((i+1)*(12+1)):
        t = input().split()
        if len(t) == 1:
            data[i][idx] = float(t[0]); idx += 1
        else:
            data[i][4+int(t[1])] += float(t[0])
    for j in range(12):
        data[i][4+j] /= (i+1)

total_time = [max(i[:4]) for i in data]
mpi_time = data[-1][:4]
thread_time = [i[4:] for i in data]

x_label = 'Number of node'
x_label_load_mpi = 'Id of node'
x_label_load_thread = 'Id of thread'
title = 'multi-node, hybrid'
title_load_mpi = 'multi-node, hybrid, each node'
title_load_thread = 'multi-node, hybrid, each thread'

plot_speed_up(total_time, 4, base, x_label, title)
plot_load_balancing(mpi_time, 4, x_label_load_mpi, title_load_mpi)
plot_load_balancing_multi(thread_time, 12, x_label_load_thread, title_load_thread)
