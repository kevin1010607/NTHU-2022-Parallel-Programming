import matplotlib.pyplot as plt

def plot_time_profile(cpu, comm, io, N, x_label, title):
    x = range(1, N+1)
    plt.figure()
    plt.bar(x, cpu, color='blue', label='CPU')
    plt.bar(x, comm, color='red', label='Comm', bottom=cpu)
    plt.bar(x, io, color='green', label='IO', bottom=[i+j for i, j in zip(cpu, comm)])
    plt.legend(loc='upper right')
    plt.xticks(x)
    plt.xlabel(x_label)
    plt.ylabel('Time (sec)')
    plt.title(f'Time profile ({title})')
    plt.savefig(f'./image/time_profile_{title}.png', dpi=300)
    plt.show()

def plot_speed_up(time, N, base, x_label, title):
    x = range(1, N+1)
    speedup = list(map(lambda x: base/x, time))
    plt.figure()
    plt.plot(x, speedup, color='blue', label='real', marker='.')
    # plt.plot(x, x, color='red', label='ideal', linestyle='--')
    plt.legend(loc='upper left')
    plt.xticks(x)
    plt.xlabel(x_label)
    plt.ylabel('Speed up (factor)')
    plt.title(f'Speed up ({title})')
    plt.savefig(f'./image/speed_up_{title}.png', dpi=300)
    plt.show()

# Single node (1~12 process)
data = [[0]*3 for _ in range(12)]

for i in range(12):
    for j in range((i+1)):
        s = input().split()
        for k in range(3):
            data[i][k] += float(s[k])
    for j in range(3):
        data[i][j] /= ((i+1))

cpu, comm, io = list(zip(*data))
time = [sum(i) for i in data]
base = time[0]
x_label = 'Number of process'
title = 'single-node'

plot_time_profile(cpu, comm, io, 12, x_label, title)
plot_speed_up(time, 12, base, x_label, title)

# Multi node (1~4 node, 12 process per node)
data = [[0]*3 for _ in range(4)]

for i in range(4):
    for j in range((i+1)*12):
        s = input().split()
        for k in range(3):
            data[i][k] += float(s[k])
    for j in range(3):
        data[i][j] /= ((i+1)*12)

cpu, comm, io = list(zip(*data))
time = [sum(i) for i in data]
x_label = 'Number of node (12 cores per node)'
title = 'multi-node'

plot_time_profile(cpu, comm, io, 4, x_label, title)
plot_speed_up(time, 4, base, x_label, title)
