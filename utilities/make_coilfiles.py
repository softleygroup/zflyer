import numpy as np
import os
from ConfigParser import SafeConfigParser 		# reading config file
import shutil

folder = '../data/experiment_Ar/'

allfolders = [os.path.join(folder,o) for o in os.listdir(folder) if os.path.isdir(os.path.join(folder,o))]

config = SafeConfigParser()
config.optionxform = lambda option : option 						# no processing in parser, in particular no change of capitalisation

values = np.zeros((4, 12))

for folder in allfolders:
	config.read(folder + '/config.info')									# read config
	parms = dict(config.items('PROPAGATION'))
	values[1, :] = eval(parms['ontimes'])
	values[2, :] = values[1, :] + eval(parms['durations'])
	values[3, :] = eval(parms['durations'])
	print values
	np.savetxt(folder + '_coil.txt', values.T, fmt = '%4.2f', delimiter = '\t')
	shutil.copyfile(folder + '/config.info', folder + '_config.info')


