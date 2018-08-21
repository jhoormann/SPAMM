#!/usr/bin/env python

'''
This script reads in a Model object that has been run. The "data format"
is to pickle the Model and write it to a gzipped file. It contains all of the
chains, the sample spectrum, and the templates.

This example reads this file, uncompresses and un-pickles the object,
then creates a new triangle plot.
The model object can be used as it existed in the file the wrote it.

Usage:

% read_model_run.py model.pickle.gz
'''

import os
import sys
import copy
import gzip
import optparse
import inspect
try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np

import triangle
import matplotlib.pyplot as pl

sys.path.append(os.path.abspath("../"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent

# read the name of the SPAMM pickle from the command line.
parser = optparse.OptionParser()
parser.add_option("-m", help="SPAMM model pickle file", dest="model_filename", action="store")
parser.add_option("--name", help="name to append to output files", dest="name", action="store",default="")

(opts, args) = parser.parse_args()

if opts.model_filename is None:
    #print "\nPlease specify the file to read, e.g. \n\n% {0} -m model.pickle.gz\n\n".format(sys.argv[0])
    sys.exit(1)

model = pickle.loads(gzip.open(opts.model_filename).read())

for i in range(model.total_parameter_count):
    pl.plot(model.sampler.chain[:,:,i].T)
    pl.savefig(str(i)+".pdf", format='pdf', dpi=1000)
    pl.close()

samples = model.sampler.chain[:, 600:, :].reshape((-1, model.total_parameter_count))
#samples = model.sampler.flatchain
#exit()
params = np.median(samples,axis=0)

print('med',np.median(samples,axis=0))
flux = model.model_flux(params)
#flux = model.model_flux([1200.,0,0,0,0,0,0,0,0,0,0,0,2000.])


#fig2 = pl.plot(model.data_spectrum.wavelengths,model.data_spectrum.flux)
#fig2 = pl.plot(model.model_spectrum.dispersion,model.model_spectrum.flux)
#fig2.savefig("fit_from_pickle.png")

dataflux = copy.deepcopy(model.data_spectrum.flux)
dataflux.mask = np.array([False]*np.size(dataflux))
print('s', dataflux.mask)
pl.plot(model.data_spectrum.dispersion,dataflux,linewidth=1)
pl.plot(model.model_spectrum.dispersion,flux,linewidth=1)
pl.savefig("galaxyfit"+opts.name+".pdf", format='pdf', dpi=1000)
pl.show()
pl.close()


pl.plot(model.model_spectrum.dispersion,dataflux-flux,linewidth=0.4)
#plt.savefig("galaxy.pdf", format='pdf', dpi=1000)
pl.savefig("galaxyresidual"+opts.name+".pdf", format='pdf', dpi=1000)
pl.show()
pl.close()
#plt.close()

#print('\nWriting to file')
#matr1 = np.array((model.model_spectrum.dispersion, dataflux-flux)).transpose()
#np.savetxt("Mrk590_MGII_contandFeSub.txt", matr1)
exit()
#sys.exit(0)
fig = triangle.corner(samples, labels=model.model_parameter_names())
fig.savefig("triangle_from_pickle"+opts.name+".png")
