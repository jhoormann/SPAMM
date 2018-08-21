#! /usr/bin/env python

import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import gzip
import numpy as np
import subprocess

#-----------------------------------------------------------------------------#

def read_pickle(pname):
    try:
        p_data = pickle.loads(gzip.open(pname).read())
    except UnicodeDecodeError:
        p_data = pickle.loads(gzip.open(pname).read(), encoding="latin1")

    model = p_data["model"]
    params = p_data["comp_params"]
    if pname == "model_20180627_1534.pickle.gz":
        params = {'fe_norm_2': 3.5356725072091589e-15, 
                  'fe_norm_3': 8.9351374726858118e-15, 
                  'no_templates': 3, 
                  'fe_width': 4208.055598607859, 
                  'fe_norm_1': 9.4233576501633248e-15}
    elif pname == "model_20180627_4259.pickle.gz":
        params = {'fe_norm_2': 8.68930476e-15, 
                  'fe_width': 5450,
                  'no_templates': 3, 
                  'fe_norm_1': 1.07988504e-14, 
                  'fe_norm_3': 6.91877436e-15,
                  'norm_PL': 5e-15,
                  'slope1': 2.5}
    elif pname == "model_20180711_4746.pickle.gz":
        params = {'bc_Te': 50250.0,
                  'bc_lines': 201.5,
                  'bc_loffset': 0.0,
                  'bc_logNe': 5.5,
                  'bc_lwidth': 5050.0,
                  'bc_norm': 3e-14,
                  'bc_tauBE': 1.0,
                  'broken_pl': False,
                  'fe_norm_1': 1.07988504e-14,
                  'fe_norm_2': 8.68930476e-15,
                  'fe_norm_3': 6.91877436e-15,
                  'fe_width': 5450,
                  'norm_PL': 5e-15,
                  'slope1': 2.5}
    elif pname == "model_20180711_3027.pickle.gz":
        params['fe_norm_1'] = 1.07988504e-14
        params['fe_norm_2'] = 8.68930476e-15
        params['fe_norm_3'] = 6.91877436e-15
        params['fe_width'] = 5450 

    
    return model, params

#-----------------------------------------------------------------------------#

def plot_posteriors(pdfname, samples, labels, params=None):
    num_params = np.size(samples[0,:])                                   
    pdf_pages = PdfPages(pdfname)    

    for i in range(num_params):
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111)
        
        chain = samples[:,i]
        hist,bins = np.histogram(chain, bins=100)

        maxind = np.argmax(hist)
        max_bin = bins[maxind] 
        binsize = bins[1]-bins[0]
        maxm = max_bin + binsize/2.

        med_bin = np.median(chain)
        med = med_bin + binsize/2.

        avg_bin = np.average(chain)
        avg = avg_bin + binsize/2.
        
        ax.hist(chain, bins, color="skyblue")
        
        if params is not None:
            try:
                actual = params[labels[i]]
                std = np.std(chain)
                ax.axvspan(actual-std, actual+std, facecolor="grey", alpha=0.25, label=r"1$\sigma$={:1.3e}".format(std))
                ax.axvline(params[labels[i]], color="red", linestyle="solid", linewidth=1.5, label="Actual value={:1.3e}".format(actual))
            except KeyError:
                pass
        
        xlo = actual - binsize*12
        xhi = actual + binsize*12
        
        vmin = min([med, avg, maxm])
        vmax = max([med, avg, maxm])

        if vmin <= xlo:
            xlo = vmin - binsize
        if vmax >= xhi:
            xhi = vmax + binsize

        ax.set_xlim(xlo, xhi)
        
#        ax.axvline(center, color="red", linestyle="dotted", linewidth=1.5, label="Max")
        ax.axvline(med, color="darkviolet", linestyle="--", linewidth=1.5, label="Median={:1.3e}".format(med))
        ax.axvline(avg, color="darkblue", linestyle="--", linewidth=1.5, label="Mean={:1.3e}".format(avg))
        ax.axvline(maxm, color="fuchsia", linestyle="--", linewidth=1.5, label="Maximum={:1.3e}".format(maxm))
        ax.legend(loc="best")

        ax.set_xlabel(labels[i])
        ax.set_ylabel("Posterior PDF")
        ax.set_title(labels[i])

        pdf_pages.savefig(fig)

    pdf_pages.close()
    print("Saved {}".format(pdfname))

#-----------------------------------------------------------------------------#

def plot_models(model, samples, pname, params, ymax=None, make_gif=True, step=100, only_last=False):
    data_spectrum = model.data_spectrum
    actualcolor = "deepskyblue"
    model_name = pname.split(".p")[0]
    outdir = "gifplots_" + model_name
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    actual_comps = {}
    for component in model.components:
        actual_params = [params[x] for x in component.model_parameter_names]
        actual_comp_flux = component.flux(spectrum=data_spectrum,
                                          parameters=actual_params)
        actual_comps[component.name] = actual_comp_flux

    if only_last is True:
        sample_range = [len(samples)-1]
    else:
        sample_range = range(0, len(samples), step)
    for i in sample_range:
        print("Iteration {}".format(i))
        j = 0
        for component in model.components:
            fig = plt.figure(figsize=(15,7))
            ax = fig.add_subplot(111)
            ax.plot(data_spectrum.wavelengths, actual_comps[component.name],
                    color=actualcolor, label="Actual Flux")
            
            comp_flux = component.flux(spectrum=data_spectrum,
                                       parameters=samples[i, j:j+len(component.model_parameter_names)])
            ax.plot(data_spectrum.wavelengths, comp_flux, color="deeppink", label="Model Flux")
            compmax = max(actual_comps[component.name])
            if ymax is None:
                ymax = compmax + .1*compmax
            ax.set_ylim(0, ymax)
            ax.set_title("{}, Iteration {}".format(component.name, i))
            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel(r"ergs/s/cm$^2$")
            ax.legend(loc="upper left", framealpha=0.25)
            figname = os.path.join(outdir, "{}_iter{:06d}.png".format(component.name, i))
            fig.savefig(figname)
            if only_last is True:
                print("Saved {}".format(figname))
            j += len(component.model_parameter_names)
            plt.close(fig)

        model_spectrum = model.model_flux(params=samples[i,:])
        fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)
        ax.errorbar(data_spectrum.wavelengths, data_spectrum.flux,
                    data_spectrum.flux_error, mfc=actualcolor, mec=actualcolor,
                    ecolor=actualcolor, fmt=".", zorder=-100, label="Actual Flux") 
        ax.plot(data_spectrum.wavelengths, model_spectrum, color="deeppink", label="Model Flux")
        modelmax = max(data_spectrum.flux)
        if ymax is None:
            ymax = modelmax + .1*modelmax
        ax.set_ylim(0, ymax)
        ax.set_title("Sum Of Model Components, Iteration {}".format(i))
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel(r"ergs/s/cm$^2$")
        ax.legend(loc="upper left", framealpha=0.25)
        figname = os.path.join(outdir, "model_iter{:06d}.png".format(i))
        fig.savefig(figname)
        if only_last is True:
            print("Saved {}".format(figname))
        plt.close(fig)

    if make_gif is True:
        for component in model.components:
            cname = component.name
            gifname = os.path.join(outdir, "{}.gif".format(cname))
            subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                                   os.path.join(outdir, "{}*png".format(cname)), 
                                   gifname])
            print("Saved {}".format(gifname))
        gifname = os.path.join(outdir, "{}.gif".format(model_name))
        subprocess.check_call(["convert", "-delay", "15", "-loop", "1", 
                               os.path.join(outdir, "model*png".format(model_name)), 
                               gifname])
        print("Saved {}".format(gifname))
        

#-----------------------------------------------------------------------------#

def make_plots(pname, gif=False, last=False, step=100, burn=50):
    model, params = read_pickle(pname)
    try:
        samples = model.sampler.chain[:, burn:, :].reshape((-1, model.total_parameter_count))
    except TypeError:
        samples = model.sampler.chain[:, burn:, :].reshape((-1, 10))
        #samples = model.sampler.chain[:, burn:, :].reshape((-1, 16))
    pdfname = "{}_posterior.pdf".format(pname)
    plot_posteriors(pdfname, samples, model.model_parameter_names(), params)
    if gif is True:
        plot_models(model, samples, pname, params, step=step, only_last=last)

#-----------------------------------------------------------------------------#

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pname", help="SPAMM model pickle file", type=str)
    parser.add_argument("--gif", dest="gif", action="store_true", default=False,
                        help="Switch to make plots to create gif")
    parser.add_argument("--last", dest="last", action="store_true", default=False,
                        help="Switch to only plot last model iteration")
    parser.add_argument("--step", dest="step", default=100,
                        help="Step size for plotting chain iterations")
    args = parser.parse_args()

    make_plots(args.pname, args.gif, args.last, int(args.step))
