#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import gzip
import dill as pickle

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pl
from astropy import units

import triangle

sys.path.append(os.path.abspath("../"))

from spamm.Spectrum import Spectrum
from spamm.Model import Model
from spamm.components.NuclearContinuumComponent import NuclearContinuumComponent
from spamm.components.HostGalaxyComponent import HostGalaxyComponent
from spamm.components.FeComponent import FeComponent
from spamm.components.BalmerContinuumCombined import BalmerCombined
from spamm.components.ReddeningLaw import Extinction
from spamm.components.MaskingComponent import Mask
# TODO: astropy units for spectrum

#emcee parameters
n_walkers = 30
n_iterations = 500

# Use MPI to distribute the computations
MPI = True 

#Select your component options
# PL = nuclear continuum
# HOST = host galaxy
# FE = Fe forest

#For the moment we have only implemented individual components
#Fe and host galaxy components need more work - see tickets
#To do: implement combined - Gisella - see tickets

PL = True#False#
HOST = False
FE = False#True#
BC =  False#True#
BpC = False#True#
Calzetti_ext = False#True#
SMC_ext = False
MW_ext = False
AGN_ext = False
LMC_ext = False
maskType="Continuum"#"Emission lines reduced"#None#

show_plots = False

# ----------------
# Read in spectrum
# ----------------

if PL:
    datafile = "../Data/FakeData/PLcompOnly/fakepowlaw1_werr.dat"

if HOST:
    datafile = "../Data/FakeData/fake_host_spectrum.dat"

if FE:
    #datafile = "../Data/FakeData/for_gisella/fake_host_spectrum.dat"
    datafile = "../Data/FakeData/Iron_comp/fakeFe1_deg.dat"
    #datafile = "../Fe_templates/FeSimdata_BevWills_0p05.dat"

if BC:
    datafile = "../Data/FakeData/BaC_comp/FakeBac01_deg.dat"
if BC and BpC:
    datafile = "../Data/FakeData/BaC_comp/FakeBac_lines01_deg.dat"


# do you think there will be any way to open generic fits file and you specify hdu, npix, midpix, wavelength stuff
wavelengths, flux, flux_err = np.loadtxt(datafile, unpack=True)
mask = Mask(wavelengths=wavelengths,maskType=maskType)
spectrum = Spectrum.from_array(flux, uncertainty=flux_err, mask=mask)
#spectrum = Spectrum(maskType="Emission lines reduced")#"Cont+Fe")#
spectrum.mask=mask
spectrum.dispersion = wavelengths#*units.angstrom
spectrum.flux_error = flux_err    
pl.plot(spectrum.wavelengths,spectrum.flux)
pl.show()
#exit()

# ------------
# Initialize model
# ------------
model = Model()
model.print_parameters = False#True#False#

# -----------------
# Initialize components
# -----------------
if PL:
    nuclear_comp = NuclearContinuumComponent()
    model.components.append(nuclear_comp)

if FE:
    fe_comp = FeComponent()
    model.components.append(fe_comp)

if HOST:
    host_galaxy_comp = HostGalaxyComponent()    
    model.components.append(host_galaxy_comp)

if BC or BpC:
    balmer_comp = BalmerCombined(BalmerContinuum=BC, BalmerPseudocContinuum=BpC)
    model.components.append(balmer_comp)
    
if Calzetti_ext or SMC_ext or MW_ext or AGN_ext or LMC_ext:
    ext_comp = Extinction(MW=MW_ext,AGN=AGN_ext,LMC=LMC_ext,SMC=SMC_ext, Calzetti=Calzetti_ext)
    model.components.append(ext_comp)

model.data_spectrum = spectrum # add data
# ------------
# Run MCMC
# ------------
model.run_mcmc(n_walkers=n_walkers, n_iterations=n_iterations)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(model.sampler.acceptance_fraction)))


# -------------
# save chains & model
# ------------
with gzip.open('model.pickle.gz', 'wb') as model_output:
    model_output.write(pickle.dumps(model))

