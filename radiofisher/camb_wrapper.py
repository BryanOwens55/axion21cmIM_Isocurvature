#!/usr/bin/python
"""
Wrapper for calling CAMB from inside Python programs.
"""
import os
import re
import subprocess
import sys

C = 3e5  # km/s


def camb_params(params_fname,
                output_root='test', get_scalar_cls='F',
                get_vector_cls='F', get_tensor_cls='F', get_transfer='T',
                do_lensing='F', do_nonlinear=0, l_max_scalar=2200,
                l_max_tensor=1500, k_eta_max_tensor=3000, use_physical='T',
                ombh2=0.0224, omch2=0.12, omnuh2=0, omk=0, hubble=67.4,
                axion_isocurvature='F', alpha_ax=0, Hinf=13.7,
                w=-1, use_axfrac='F',
                omaxh2=0.00000001, m_ax=1.e-27,
                cs2_lam=1, temp_cmb=2.726, helium_fraction=0.24,
                massless_neutrinos=3.046, massive_neutrinos=0,
                nu_mass_eigenstates=1, nu_mass_fractions=1, share_delta_neff='T',
                initial_power_num=1, pivot_scalar=0.05,
                pivot_tensor=0.05, scalar_amp__1___=2.3e-9,
                scalar_spectral_index__1___=0.965, scalar_nrun__1___=0,
                tensor_spectral_index__1___=0,
                initial_ratio__1___=0, tens_ratio=0,
                reionization='T', re_use_optical_depth='T',
                re_optical_depth=0.06, # to be consistent when combining with CMB surveys
                re_redshift=11, re_delta_redshift=1.5, re_ionization_frac=-1,
                RECFAST_fudge=1.14, RECFAST_fudge_He=0.86, RECFAST_Heswitch=6,
                RECFAST_Hswitch='T', initial_condition=1,
                initial_vector='-1 0 0 0 0', vector_mode=0, COBE_normalize='F',
                CMB_outputscale=7.4311e12, transfer_high_precision='F',
                transfer_kmax=2, transfer_k_per_logint=0,
                transfer_num_redshifts=1, transfer_interp_matterpower='T',
                transfer_redshift__1___=0, transfer_filename__1___='transfer_out.dat',
                transfer_matterpower__1___='matterpower.dat',
                scalar_output_file='scalCls.dat', vector_output_file='vecCls.dat',
                tensor_output_file='tensCls.dat', total_output_file='totCls.dat',
                lensed_output_file='lensedCls.dat',
                lensed_total_output_file='lensedtotCls.dat',
                lens_potential_output_file='lenspotentialCls.dat',
                FITS_filename='scalCls.fits', do_lensing_bispectrum='F',
                do_primordial_bispectrum='F', bispectrum_nfields=1,
                bispectrum_slice_base_L=0, bispectrum_ndelta=3,
                bispectrum_delta__1___=0, bispectrum_delta__2___=2,
                bispectrum_delta__3___=4, bispectrum_do_fisher='F',
                bispectrum_fisher_noise=0, bispectrum_fisher_noise_pol=0,
                bispectrum_fisher_fwhm_arcmin=7,
                bispectrum_full_output_file='',
                bispectrum_full_output_sparse='F',
                feedback_level=1, lensing_method=1, accurate_BB='F', massive_nu_approx=1,
                accurate_polarization='T', accurate_reionization='T',
                do_tensor_neutrinos='T', do_late_rad_truncation='T',
                number_of_threads=0, high_accuracy_default='F',
                accuracy_boost=1, l_accuracy_boost=1, l_sample_boost=1):
    """
    Define a dictionary of all parameters in CAMB, set to their default values.
    """
    
    # Get dict. of arguments 
    args = locals()

    # Get all parameters into the CAMB param.ini format
    camb_params_text = ""
    for key in args:
        keyname = key
        if "__" in key:  # Rename array parameters
            keyname = key.replace("___", ")").replace("__", "(")
        line_str = "=".join((keyname, str(args[key])))
        camb_params_text += line_str + "\n"

    # Output params file
    print("Writing parameters to", params_fname)
    f = open("paramfiles/" + params_fname, 'w')
    f.write(camb_params_text)
    f.close()


def run_camb(params_fname, camb_exec_dir, params_fname2=''):
    """
    Run (axion)CAMB, using a given (pre-written) params file (see camb_params). Waits
    for (axion)CAMB to finish before returning. Returns a dictionary of derived values
    output by CAMB to stdout.
    """



    # Change directory and call CAMB
    cwd = os.getcwd()
    os.chdir(camb_exec_dir)

    
    #run isocurvature version
    params_path2 = cwd + "/paramfiles/" + params_fname2
    print("Running CAMB on", params_path2)
    output2 = subprocess.check_output(["./camb", params_path2]).decode(sys.stdout.encoding)
    print('-' * 25 + ' axionCAMB Isocurvature output ' + '-' * 25)
    print(output2)
    print('-' * 50)
    # Capture on-screen output of derived parameters
    vals2 = {}
    for line in output2.split("\n"):
        # Special cases: sigma8 and tau_recomb
        if "sigma8" in line:
            s8line2 = line[line.find('sigma8'):]  # Only get sigma8 part of string
            match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
            vals2['sigma8'] = float(re.findall(match_number, s8line2)[1])
        elif "tau_recomb" in line:
            tau2 = re.findall(r'\b\d+.\d+\b', line)
            vals2['tau_recomb/Mpc'] = float(tau2[0])
            vals2['tau_now/Mpc'] = float(tau2[1])
        elif "z_EQ" in line:
            vals2['z_EQ'] = float(re.findall(r'\b\d+.\d+\b', line)[0])
        else:
            # All other params can just be stuffed into a dictionary
            try:
                key2, val2 = line.split("=")
                vals2[key.strip()] = float(val2)
            except:
                pass
    
    
    params_path = cwd + "/paramfiles/" + params_fname
    print("Running CAMB on", params_path)
    output = subprocess.check_output(["./camb", params_path]).decode(sys.stdout.encoding)
    print('-' * 25 + ' axionCAMB output ' + '-' * 25)
    print(output)
    print('-' * 50)
    # Capture on-screen output of derived parameters
    vals = {}
    for line in output.split("\n"):
        # Special cases: sigma8 and tau_recomb
        if "sigma8" in line:
            s8line = line[line.find('sigma8'):]  # Only get sigma8 part of string
            print('\n\n')
            print(re.findall(r'\b\d+.\d+\b', s8line))
            print(s8line)
            print('\n\n')
            match_number = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
            vals['sigma8'] = float(re.findall(match_number, s8line)[1])
        elif "tau_recomb" in line:
            tau = re.findall(r'\b\d+.\d+\b', line)
            vals['tau_recomb/Mpc'] = float(tau[0])
            vals['tau_now/Mpc'] = float(tau[1])
        elif "z_EQ" in line:
            vals['z_EQ'] = float(re.findall(r'\b\d+.\d+\b', line)[0])
        else:
            # All other params can just be stuffed into a dictionary
            try:
                key, val = line.split("=")
                vals[key.strip()] = float(val)
            except:
                pass
    


    # Change back to the original directory
    os.chdir(cwd)
    return vals
