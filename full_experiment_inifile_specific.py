"""
Calculate C_ell constraints (and Fisher matrix) for all redshift bins for a given
experiment.
"""
import os
import sys
import numpy as np
sys.path.append('radiofisher/')
sys.path.append('halo_model/')
import radiofisher as rf
from radiofisher import experiments
from mpi4py import MPI

comm = MPI.COMM_WORLD
myid = comm.Get_rank()
size = comm.Get_size()

verbos = 0
################################################################################
# Set-up experiment parameters
################################################################################
if verbos >= 1 and myid == 0:
    print('#' * 50)
    print('Set-up experiment parameters')
    print('#' * 50)

# Load cosmology and experimental settings
e = experiments

# Take command-line argument for which survey to calculate, or set manually
if len(sys.argv) > 1:
    input_filename = str(sys.argv[1])
else:
    print('No input file specified. Abort.')
    raise IOError

with open(input_filename, 'r') as i:
    lines = i.readlines()

expt_dic, analysis_dic, cosmo_dic = rf.load_cosmology_input(file=lines)
print(analysis_dic)

# Label experiments with different settings
EXPT_LABEL = '_%s_%s-noise' % (expt_dic['mode'], analysis_dic['noise'])

if cosmo_dic['mnu'] > 1.e-6:
    EXPT_LABEL += '_massive-neutrinos'
EXPT_LABEL += '_ma%i_fiducial-axfrac%s' % (cosmo_dic['ma'], str(cosmo_dic['axion_fraction']))

expt_list = [
    ('SKA1MID',     e.SKA1MIDfull),     # 0
    ('BINGO',       e.BINGO),           # 1
    ('MeerKATb1',   e.MeerKATb1),       # 2
    ('HIRAX',       e.HIRAX),           # 3
    ('CV1', 	    e.CVlimited_z0to3), # 4
    ('CV2',         e.CVlimited_z0to5), # 5
    ('exptS',       e.exptS),           # 6
    ('aexptM',      e.exptM),           # 7
    ('exptL',       e.exptL),           # 8
    ('GBT',         e.GBT),             # 9
    ('Parkes',      e.Parkes),          # 10
    ('GMRT',        e.GMRT),            # 11
    ('FAST',        e.FAST),            # 12
]
names, expts = zip(*expt_list)
names = list(names)
expts = list(expts)

################################################################################

names[expt_dic['k']] += EXPT_LABEL
if myid == 0:
    print("=" * 50)
    print("Survey:", names[expt_dic['k']])
    print("=" * 50)

# Set settings depending on chosen experiment
expt = expts[expt_dic['k']]
expt['mode'] = expt_dic['mode']
if verbos >= 1 and myid == 0: print('Set survey parameters: \n '
                                    'ttot = %s hrs \n ' % expt_dic['ttot'])
expt['ttot'] *= expt_dic['ttot'] / 1e4

# Load n(u) interpolation function, if needed
if (expt['mode'][0] == 'i') and 'n(x)' in expt.keys():
    expt['n(x)_file'] = expt['n(x)']
    expt['n(x)'] = rf.load_interferom_file(expt['n(x)'])

directory = 'output/ma%i' % cosmo_dic['ma']
directory += '/axfrac%s' % str(cosmo_dic['axion_fraction']).replace('.', 'p')
if myid == 0:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    created_directories = True
else:
    created_directories = None
created_directories = comm.bcast(created_directories, root=0)

survey_name = names[expt_dic['k']]
root = directory + '/' + survey_name

# Define redshift bins
if verbos >= 1 and myid == 0: print('Defining redshift bins...')
zs, zc = rf.zbins_const_dz(expt, dz=0.05) #
if verbos >= 1 and myid == 0: print('Number of redshift bins = %i' % len(zs))
if verbos >= 1 and myid == 0: print('Done.')

print("*" * 50)
for key in cosmo_dic.keys():
    print("%20s: %s" % (key, cosmo_dic[key]))
print("*" * 50)
for key in expt.keys():
    print("%20s: %s" % (key, expt[key]))
print("*" * 50)

################################################################################
# Loop through redshift bins, assigning them to each process
################################################################################
if verbos >= 1 and myid == 0:
    print('#' * 50)
    print('Loop through redshift bins, assigning them to each process')
    print('#' * 50)

i=0

print(">>> %2d working on redshift bin %2d -- z = %3.5f" % (myid, i, zc[i]))
# Add z to cosmological library in order to pass this value to axionCAMB input
cosmo_dic['z'] = zc[i]

# Set filename of fiducial parameters for cached power spectrum dictionary.
fname_PS = 'PS_dictionaries/cache_powerspec_ma{}_fiducial-axfrac{}_z{}.npy'.format(cosmo_dic['ma'],
                                                                                    cosmo_dic['axion_fraction'],
                                                                                    zc[i])

# Calculate basic Fisher matrix for C_ell

for key, value in cosmo_dic.items() :
    print (key)


lst_powerspec = []
lst_powerspec2 = []
lst3 = []

num = 1
for j in range(num):

    print('\n\n\n')
    print(str(round(j/num*100)) + '%')
    print('\n\n\n')

    if verbos >= 1: print('Calculate Fisher matrix for C_ell.')
    '''
    F, paramnames, deltacl_arr, Cl, ell_arr, derivs, powerspec_dic, k, P_HI = rf.fisher_Cl(zmin=zs[i], zmax=zs[i + 1], cosmo=cosmo_dic, expt=expt,
                                                           cachefile=fname_PS, analysis_specifications=analysis_dic,
                                                            survey_name=survey_name.replace(EXPT_LABEL, ''))
    '''

    from fisher_P_HI import fisher_Cl
    cosmo_dic['z'] = 0.0
    F, paramnames, deltacl_arr, Cl, ell_arr, derivs, powerspec_dic, k, P_HI = fisher_Cl(0.0,0.001, cosmo=cosmo_dic, expt=expt,
                                                           cachefile=fname_PS, analysis_specifications=analysis_dic,
                                                            survey_name=survey_name.replace(EXPT_LABEL, ''))



    lst_powerspec.append(powerspec_dic['PS_total'])
    lst_powerspec2.append(powerspec_dic['PS_axion+baryon+CDM'])
    lst3.append(P_HI)
lst3.append(k)

# Save Fisher matrix and k bins

'''
if verbos >= 1: print('Save diagonal of Fisher matrix and Delta C_ell and N_ell.')
np.savetxt(root + "-fisher-Cl-full_%d.dat" % i, F, header=" ".join(paramnames))
np.savetxt(root + "-deltacl_%i.dat" % i, deltacl_arr)
np.savetxt(root + "-Cl_%i,%2d.dat" % (i,zc[i]), Cl)
np.savetxt('output/derivs/' + "derivs_%i,%3.5f.dat" % (i,zc[i]), np.transpose(derivs), header=" ".join(paramnames))
np.savetxt('output/PS/' + "PS_tot_%i,%3.5f.dat" % (i,zc[i]), np.transpose(powerspec_dic['PS_total']))
np.savetxt('output/PS/' + "PS_%i,%3.5f.dat" % (i,zc[i]), np.transpose(powerspec_dic['PS_axion+baryon+CDM']))
np.savetxt('output/PS/' + "k_%i,%3.5f.dat" % (i,zc[i]), np.transpose(powerspec_dic['k']))
'''

np.savetxt('output/powerspecs/' + "PS_%i,%3.5f.dat" % (i,zc[i]), np.transpose(lst_powerspec))
np.savetxt('output/powerspecs/' + "PS_2_%i,%3.5f.dat" % (i,zc[i]), np.transpose(lst_powerspec2))
np.savetxt('output/powerspecs/' + "P_HI%i.dat" % (cosmo_dic['ma']), np.transpose(lst3))

#PS_total




if myid == 0:
    if verbos >= 1: print('... and save center of k bins.')
    np.savetxt(root + "-ell.dat", ell_arr)
    np.savetxt(root + "-redshift_%i.dat" % i, zc)

if verbos >= 1: print(">>> %2d finished redshift bin %2d -- z = %3.3f" % (myid, i, zc[i]))
del F, paramnames, deltacl_arr, ell_arr

comm.barrier()
if myid == 0: print("Finished.")


'''
k
transfer_total
growth_rate
transfer_CDM
transfer_baryon
transfer_axion
transfer_massless-neutrino
transfer_massive-neutrino
transfer_axion+baryon+CDM
PS_axion+baryon+CDM
PS_total
omega_bh2
omega_ch2
omega_nuh2
omega_k
hubble
omega_axh2
k_pivot
initial_amplitude
scalar_index
transfer_kmax
transfer_interp_matterpower
z
transfer_filename
matterpower_filename
number_of_k_values
omega_axion+baryon+CDM_0
hash
'''
