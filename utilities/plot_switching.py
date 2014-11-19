from matplotlib import pyplot as plt
import numpy as np

selection = [1, 2, 9]

with open('collected_times', 'rb') as ifile:
	lines = ifile.readlines()

for s in selection:
	label = lines[4*(s - 1)]
	on = eval(lines[4*(s-1) + 2][10:])
	dur = eval(lines[4*(s-1) + 3][12:])
	off = on + dur
	plt.figure(0)
	plt.plot(on, label=label)
	plt.figure(1)
	plt.plot(dur, label=label)

plt.figure(0)
plt.legend(loc=2)
plt.figure(1)
plt.legend(loc=4)
plt.show()