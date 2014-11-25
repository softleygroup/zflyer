from matplotlib import pyplot as plt
import numpy as np
import os

folder = '../data/experiment/'
files = [f for f in os.listdir(folder) if f.endswith('coil.txt')]

default = np.genfromtxt(folder + '480_default_coil.txt')

for f in files:
	data = np.genfromtxt(folder + f, delimiter = '\t')
	plt.figure(0)
	plt.plot(data[:, 1] - default[:, 1], label=f)
	plt.figure(1)
	plt.plot(data[:, 2] - default[:, 2], label=f)
	plt.figure(2)
	plt.plot(data[:, 3] - default[:, 3], label=f)


plt.figure(0)
plt.title('on times')
plt.legend(loc=2)
plt.figure(1)
plt.title('off times')
plt.legend(loc=4)
plt.figure(2)
plt.title('durations')
plt.legend(loc=4)

plt.show()
raise RuntimeError

selection = [0, 1, 2, 3, 4, 5, 6]

with open('../data/collected_times', 'rb') as ifile:
	lines = ifile.readlines()

for s in selection:
	label = lines[4*(s - 1)]
	on = eval(lines[4*(s-1) + 2][10:])
	dur = eval(lines[4*(s-1) + 3][12:])
	off = on + dur
	plt.figure(0)
	plt.plot(on, label=label[3:])
	plt.figure(1)
	plt.plot(dur, label=label[3:])
	plt.figure(2)
	plt.plot(on+dur, label=label[3:])


plt.show()

