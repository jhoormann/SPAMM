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
import matplotlib.pyplot as pl
from astropy import units

import warnings
from astropy.io import fits
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from pysynphot import observation
    from pysynphot import spectrum as pysynspec

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

def wavelengthobs(cube,ext):
        """Define wavelength solution."""
        hdulist=fits.open(cube)
        crpix=hdulist[1].header['CRPIX1']-1.0 # Central pixel value. The -1.0 is needed as Python is ZERO indexed
        crval=hdulist[1].header['CRVAL1'] # central wavelength value
        cdelt=hdulist[1].header['CDELT1'] # Wavelength interval between subsequent pixels
        n_pix = hdulist[1].header["NAXIS1"]
        wave=((np.arange(n_pix)-crpix)*cdelt)+crval
        return wave



#emcee parameters
n_walkers = 50
n_iterations = 1500

# Use MPI to distribute the computations
MPI = False

#Select your component options
# PL = nuclear continuum
# HOST = host galaxy
# FE = Fe forest

#For the moment we have only implemented individual components
#Fe and host galaxy components need more work - see tickets
#To do: implement combined - Gisella - see tickets

PL = True#False#
HOST = False#True#
FE = True#False#
BC =  False#True#
BpC = False#True#
Calzetti_ext = False#True#
SMC_ext = False
MW_ext = False
AGN_ext = True#False
LMC_ext = False
maskType="Emission lines reduced"#"Continuum"#"Emission lines complete"#None#

show_plots = False

# ----------------
# Read in spectrum
# ----------------
datafile = "2939999066_scaled.fits"
hdulist=fits.open(datafile)
n = len(hdulist)//3 -1
crpix=hdulist[2].data
flux=hdulist[2].data
wave1 = wavelengthobs(datafile,0)
wavelengths = wave1[150:]
z =  1.57774
wavelengths/=1.+z
print('wave',np.min(wavelengths),np.max(wavelengths))
for jj in range(n):
    ind = jj*3+3
    flux1 = hdulist[ind].data
    flux1var = hdulist[ind+1].data

    # do you think there will be any way to open generic fits file and you specify hdu, npix, midpix, wavelength stuff

    flux = flux1[150:]
    print('npnanmaxflux',np.nanmax(flux))
    print('maxflux',max(flux))
    flux_err = np.sqrt(flux1var[150:])

    mask = Mask(wavelengths=wavelengths,maskType=maskType)
    flux_err /=np.nanmedian(flux)
    flux /=np.nanmedian(flux)
    spectrum = Spectrum(0)
    spectrum.flux=flux
    spectrum.wavelengths = wavelengths
    spectrum.mask=mask
    spectrum.flux_error = flux_err
    if jj == 0:
        pl.plot(wavelengths,flux)
        pl.plot(spectrum.wavelengths,spectrum.flux)
        pl.show()


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
        fe_comp = FeComponent(spectrum)
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
    with gzip.open('model.pickleozdes'+str(jj)+'.gz', 'wb') as model_output:
        model_output.write(pickle.dumps(model))
