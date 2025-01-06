import sys

sys.path.append('/home/users/kokron/Libraries/velocileptors/velocileptors') #your lakelag directory here
sys.path.append('/home/users/kokron/Libraries/velocileptors') #your lakelag directory here
sys.path.append('/home/users/kokron/Libraries/') #your lakelag directory here

import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import math
import time
import LPT.cleft_fftw as cleft
import gc
import pmesh
from simplehod import hod, mkhodp, mknint, mksat, mkcen
from nbodykit.lab import *
from scipy.special import erf
from mpi4py import MPI
import illustris_python as il
import matplotlib.gridspec as gridspec
from glob import glob
import os

# some plotting stuff
plt.rcParams['figure.dpi'] = 85
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['legend.fontsize'] = 16

# define some global variables
comm = MPI.COMM_WORLD # a predefined intracommunicator instance
rank = comm.Get_rank() # the calling process rank
st = time.time()
nranks  = comm.Get_size() # the number of processes in a communicator 
home = '/oak/stanford/orgs/kipac/users/mahlet/bias/data/'
invmatvec = []
cuts = {0: {'high': {'TNG': [-10.211, 10.156], 'UM': [-10.121, 10.316]}, 
         'low': {'TNG': [-11.664, 10.81], 'UM': [-10.713, 10.889]}, 
         'medium':{'TNG': [-11.254, 10.396], 'UM': [-10.253, 10.534]}}, 
        0.5: {'high': {'TNG': [-10.394, 10.108], 'UM': [-11.156, 10.3]}, 
         'low': {'TNG': [-11.68, 10.758], 'UM': [-11.488, 10.829]}, 
         'medium':{'TNG': [-11.46, 10.372], 'UM': [-11.31, 10.509]}}, 
        1: {'high': {'TNG': [-9.591, 9.998], 'UM': [-9.846, 10.183]}, 
         'low': {'TNG': [-10.86, 10.688], 'UM': [-11.337, 10.763]}, 
         'medium':{'TNG': [-10.172, 10.286], 'UM': [-11.02, 10.426]}},
        1.5: {'high': {'TNG': [-9.253, 9.862], 'UM': [-9.141, 9.863]}, 
         'low': {'TNG': [-10.339, 10.606], 'UM': [-10.964, 10.665]}, 
         'medium':{'TNG': [-9.4, 10.145], 'UM': [-9.421, 10.213]}}}
    
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def MPI_mean(var):
    '''
    Computes the mean of a variable that is held across multiple processes.
    '''
    
    procsum = var*np.ones(1)
    recvbuf = None
    
    # Create the rank 0 array that will receive all of the nbar sums
    if rank==0:
        recvbuf = np.zeros(shape=[nranks,1])
        
    # Pull in all of the nbar values
    comm.Gather(procsum, recvbuf, root=0)

    # Compute the mean in rank 0
    if rank==0:
        
    # mean here is the sum of recvbuf (which holds all of the vars) across ranks.
        varmean = np.ones(1)*np.sum(recvbuf)/nranks
    else:
        
    # Give a random variable to the other ranks before broadcasting back
        varmean = np.ones(1)
    comm.Bcast(varmean, root=0)
    
    return varmean[0]

def MPI_sum(var): 
    
    '''
    Computes the sum of a variable that is held across multiple processes.
    '''
    
    procsum = var*np.ones(1, dtype='complex64')
    recvbuf = None
    
    #Create the rank 0 array that will receive all of the field sums
    if rank==0:
        recvbuf = np.zeros(shape=[nranks,1], dtype='complex64')
        
    #Pull in all of the sum values
    comm.Gather(procsum, recvbuf, root=0)

    #Compute the mean in rank 0
    if rank==0:
        
    #sum here is the sum of recvbuf (which holds all of the vars) across ranks.
        varsum = np.ones(1, dtype='complex64')*np.sum(recvbuf)
    else:
        
    #Give a random variable to the other ranks before broadcasting back
        varsum = np.ones(1, dtype='complex64')
    comm.Bcast(varsum, root=0)
    return varsum[0]

def zhai_cen(M, log10M_min, sigma_log, fcen=1):
    diffs = np.log10(M) - log10M_min


    return fcen*0.5 * (1 + erf(diffs/sigma_log))

def CompensateCICAliasing(w, v):
    """
    Return the Fourier-space kernel that accounts for the convolution of
        the gridded field with the CIC window function in configuration space,
            as well as the approximate aliasing correction
    From the nbodykit documentation.
    """
    
    for i in range(3):
        wi = w[i]
        v = v / (1 - 2. / 3 * np.sin(0.5 * wi) ** 2) ** 0.5
    return v

def zhai_sat(M, Msat, Mcut, alpha, Ncen):
    exps = np.exp(-Mcut/M)
    powerlaw = (M/Msat)**alpha

    return exps*powerlaw*Ncen

def mpiprint(text):
    if rank==0:
        print(text)
        sys.stdout.flush()
    else:
        pass

def jdr_sat(M, log10M_min, sigma_log, M0, M1, alpha,fcen=1):
    powerlaw = ((M - M0)/M1)**alpha
    diffs = np.log10(M) - log10M_min

    censpart = fcen*0.5 * (1 + erf(diffs/sigma_log))

    mask = np.where(M < M0)[0]

    powerlaw.iloc[mask] = 0

    return powerlaw*censpart

def zhou_sat(M, log10M_min, sigma_log, M0, M1, alpha):
    powerlaw = ((M - M0)/M1)**alpha
    diffs = np.log10(M) - log10M_min

    censpart = 0.5 * (1 + erf(diffs/sigma_log))

    mask = np.where(M < M0)[0]

    powerlaw.iloc[mask] = 0

    return powerlaw

def fieldread_mpi(array, pm, nmesh): #rank
    
    '''
    Loads in a (pre-chunked) numpy array into an nbodykit complex field in MPI-parallel operations.

    Inputs:
    array - the field in question, [N, N, N]
    pm - the particlemesh object 
    rank - the MPI rank of this process

    Outputs:
    f - realfield locally held in rank
    knorms - array of |k| that will be hosted in this rank.
    '''
    f = pmesh.pm.RealField(pm)

    Nx, Ny, Nz = nmesh//np.array(f.shape) # gets the factor that each dimension is reduced by when chunking
    
    print(np.shape(array)) #250
    print(np.shape(f[...])) #313
    f[...] = array
    print(np.shape(f))

    return f, [Nx, Ny, Nz]

def fieldread_mpi_original(array, pm, nmesh): #rank
    '''
    Loads in a(n unchunked) numpy array into an nbodykit complex field in MPI-parallel operations.

    Inputs:
    array - the field in question, [N, N, N]
    pm - the particlemesh object 
    rank - the MPI rank of this process

    Outputs:
    f - realfield locally held in rank
    knorms - array of |k| that will be hosted in this rank.
    '''
    
    f = pmesh.pm.RealField(pm)

    Nx, Ny, Nz = [round(i) for i in nmesh/np.array(f.shape)] # gets the factor that each dimension is reduced by when chunking
    print('f.shape', f.shape, 'rank:', rank)
    print('nmesh: %i'%(nmesh), 'rank:', rank)
    print('Nx, Ny, Nz: %i %i %i'%(Nx, Ny, Nz), 'rank:', rank)

    Ni = rank//Ny
    print('Ni:', Ni, 'rank:', rank)

    Nj = (rank)%Ny
    print('Nj:', Nj, 'rank:', rank)
    
    x0 = int(np.floor(Ni*nmesh/Nx+Ni))
    x1 = int(np.floor((Ni+1)*nmesh/Nx+Ni)+1)
    y0 = int(np.floor(Nj*nmesh/Ny+Nj))
    y1 = int(np.floor((Nj+1)*nmesh/Ny+Nj)+1)
    
    print('array:', np.shape(array), 'rank:', rank)
    print('array chunk:', np.shape(array[x0:x1, y0:y1,:]), 'rank:', rank)

    f[...] = array[x0:x1, y0:y1,:] # manually chunk the box+save it to the chunk we're on
    
    return f, [Nx, Ny, Nz]

def fieldknorm(field):
    '''
    Gets array of knorms that are in this pencil. 
    Assumes the FFT'd array has been transposed.
    Input:
        -field: TransposedComplexField from pmesh. If MPI, this will give
        only the pencil associated to this rank.
    Output:
        -knorm: array of |k| for every entry of this field.
    '''
    
    #Transpose operation permutes 123->312 for array dimensions

    kxs = field.x[0].flatten()
    kys = field.x[1].flatten()
    kzs = field.x[2].flatten()

    kx, ky, kz = np.meshgrid(kxs, kys, kzs, indexing='ij')
    
    knorm = np.sqrt(kx**2 + ky**2 + kz**2)
  
    return knorm

def halocat_to_cut(snapshot, Lbox, color, hodtype, density, z):
    '''
    Takes TNG halocat and returns a galaxy cut
    Input:
        -snapshot: Illustris TNG snapshot number.
        -Lbox: the size of the box in Mpc/h.
        -color: red (quenched) or blue (star-forming) cut.
    Output:
        -nbk_cat: nbodykit arraycatalog object
    '''
    
    mpiprint(hodtype)
    if hodtype == 'TNG':
    
        # load galaxy catalogues in TNG
        basePath = '/oak/stanford/orgs/kipac/users/mahlet/TNG300-1/output'
        fields   = ['SubhaloMassType','SubhaloSFR', 'SubhaloPos', 'SubhaloFlag']
        subhalos = il.groupcat.loadSubhalos(basePath, snapshot, fields = fields)
        fields_tng = [subhalos[i] for i in fields] # ckpc/h --> cMpc/h
        sm_zero, sfr_zero, pos = [field[fields_tng[-1]] for field in fields_tng[:-1]] 
        sm_zero = sm_zero[:,4]*10**10
        pos /= 1000
        mpiprint(fields[:-1])

    if hodtype == 'UM':
        
        fdir = '/oak/stanford/orgs/kipac/users/chto/buzzard_redmapper/L205n2500TNG_DM/sfrca/'
        catalogs = glob(fdir + 'sfr_catalog_*')
        lst=[]
        for cat in catalogs:
            mpiprint(cat)
            mpiprint('a = %.2f'%float(cat[-12:-4])+' z = %.2f'%(1./float(cat[-12:-4]) - 1))
            lst.append(1./float(cat[-12:-4]) - 1)

        # find the catalog closest to z = 1!
        K = min(range(len(lst)), key = lambda i: abs(lst[i]-z))

        # pick the one closest to z=1!
        dtype = np.dtype(dtype=[('id', 'i8'),('descid','i8'),('upid','i8'),
                                ('flags', 'i4'), ('uparent_dist', 'f4'),
                                ('pos', 'f4', (6)), ('vmp', 'f4'), ('lvmp', 'f4'),
                                ('mp', 'f4'), ('m', 'f4'), ('v', 'f4'), ('r', 'f4'),
                                ('rank1', 'f4'), ('rank2', 'f4'), ('ra', 'f4'),
                                ('rarank', 'f4'), ('A_UV', 'f4'), ('sm', 'f4'),
                                ('icl', 'f4'), ('sfr', 'f4'), ('obs_sm', 'f4'),
                                ('obs_sfr', 'f4'), ('obs_uv', 'f4'), ('empty', 'f4')],
                         align=True)
        halos = np.fromfile(catalogs[K], dtype=dtype)
        
        """
        Field explanations:
        **Note that halo masses are in Msun/h and stellar masses/SFRs are in Msun.
        ID: Unique halo ID
        DescID: ID of descendant halo (or -1 at z=0).
        UPID: -1 for central halos, otherwise, ID of largest parent halo
        Flags: Ignore
        Uparent_Dist: Ignore
        pos[6]: (X,Y,Z,VX,VY,VZ)
        X Y Z: halo position (comoving Mpc/h)
        VX VY VZ: halo velocity (physical peculiar km/s)
        M: Halo mass (Bryan & Norman 1998 virial mass, Msun/h)
        V: Halo vmax (physical km/s)
        MP: Halo peak historical mass (BN98 vir, Msun/h)
        VMP: Halo vmax at the time when peak mass was reached.
        R: Halo radius (BN98 vir, comoving kpc/h)
        Rank1: halo rank in Delta_vmax (see UniverseMachine paper)
        Rank2, RA, RARank: Ignore
        A_UV: UV attenuation (mag)
        SM: True stellar mass (Msun)
        ICL: True intracluster stellar mass (Msun)
        SFR: True star formation rate (Msun/yr)
        Obs_SM: observed stellar mass, including random & systematic errors (Msun)
        Obs_SFR: observed SFR, including random & systematic errors (Msun/yr)
        Obs_UV: Observed UV Magnitude (M_1500 AB)
        """

        # define variables
        sfr_zero = halos['sfr']
        sm_zero = halos['sm'] * 0.678 # let us convert from UM solar mass --> solar mass / h (where h = 0.678 according to Planck 2016)
        pos = halos['pos'][:,:3] #*0.500677 #cMpc/h --> Mpc/h --> keep it in cMpc/h actually!
    
    # now add an offset to make things run more smoothly
    mpiprint(min(sm_zero))
    offset = 1e-13
    sm = sm_zero + offset
    mpiprint(min(sm))

    # define variables
    volume = Lbox**3 # Mpc/h
    ssfr_log, sm_log = np.log10((sfr_zero + offset)/sm), np.log10(sm)

    ssfr_cut, sm_cut = cuts[z][density][hodtype]
    if color == 'red':
        ind = np.where((ssfr_log < ssfr_cut) & (sm_log > sm_cut))[0]
        cut = len(ind)/volume
        mpiprint("{:.2e}, {:.2e}".format(cut/5e-4, cut)) 

    # blue galaxies in UM
    elif color == 'blue':
        ind = np.where((ssfr_log > ssfr_cut) & (sm_log > sm_cut))[0]
        cut = len(ind)/volume
        mpiprint("{:.2e}, {:.2e}".format(cut/5e-4, cut)) 
        
    else:
        ind = np.where(sm_log > sm_cut)[0]
        cut = len(ind)/volume
        mpiprint("{:.2e}, {:.2e}".format(cut/10e-4, cut)) 
            
    mpiprint('Mean MPI nbar is %.6f?'%cut)
    
    return pos[ind], cut
    
def pk(field, title, kvals_file, lin_pk_file, res, correlation = 'auto', second = None):
        
    # compensate for interpolation
    field = field.apply(CompensateCICAliasing, kind='circular')
    mpiprint('gal_k: '+str(np.shape(field)))

    # Create the FFTPower objects
    if correlation == 'auto':
        lin_pk = FFTPower(field, '1d', kmin = 1e-5)
    elif correlation == 'cross':
        lin_pk = FFTPower(field, '1d', kmin = 1e-5, second = second)
    powervals = lin_pk.power['power'].real 
    mpiprint('lin_pk: '+str(np.shape(lin_pk)))

    # k arrays are set like this
    kvals = lin_pk.power['k'] 
    
    np.save(kvals_file+'_'+title, kvals)
    np.save(lin_pk_file+'_'+title, powervals)

    return powervals, kvals
    
def sanity_check(gal_k, density_k, Lbox, cut, color, hodtype, density, kvals_file, lin_pk_file, res):
    
    # sanity check the density contrast of galaxies... should look like visualizations
    title_gal, title_density = 'gal_k', 'density_k'
    lin_pk_gal, kvals_gal = pk(gal_k, title_gal, kvals_file, lin_pk_file, res)
    
    # load the galaxy clustering computed locally 
    kvals = np.load(kvals_file+'.npy')
    lin_pk = np.load(lin_pk_file+'.npy')
    mpiprint(np.shape(lin_pk))
    
    # Plot the P(k)    
    plt.figure(figsize=(10,8))
    grid = gridspec.GridSpec(2,1, height_ratios = [4,1])
    grid.update(hspace = 0)
    grid0 = plt.subplot(grid[0])
    grid1 = plt.subplot(grid[1])
    
    # load the galaxy clustering computed locally 
    grid0.loglog(kvals, lin_pk, label = hodtype)
    grid0.loglog(kvals_gal, lin_pk_gal, label = hodtype+', MPI', linestyle = '--') 
    mpiprint('debugging1')
    mpiprint(type(density_k))
    
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    if np.shape(density_k) != ():
        mpiprint('debugging2')
        lin_pk_density, kvals_density = pk(density_k, title_density, kvals_file, lin_pk_file, res)
        grid0.loglog(kvals_density, lin_pk_density, label = hodtype+', MPI (HEFT)', linestyle = '--')  
        grid1.semilogx(kvals, lin_pk_density/lin_pk, color = colors[2])  

    # Compare the inverse of the number density of the sample with each line
    if color == 'red':
        grid0.set_title('Quenched')
    elif color == 'blue':
        grid0.set_title('Star-Forming')
    else:
        grid0.set_title('Total')
        color = 'purple'
    grid0.axhline(1/cut, color = color, ls = "--", linewidth = 2.5, alpha = 0.5)
    grid0.set_xlim(kvals[0], kvals[-1])
    grid0.set_ylim(7.25, 3.25e4)
    grid0.set_ylabel('P(k)')
    grid0.legend()
    
    grid1.axhline(1, color = 'grey', linewidth = 2.5, linestyle = '--')
    grid1.semilogx(kvals, lin_pk_gal/lin_pk, color = colors[1])  
    grid1.set_xlabel('k')
    grid1.set_xlim(kvals[0], kvals[-1])
    grid1.set_ylabel('Ratio')
    grid1.minorticks_on()
    grid1.set_ylim(0.55, 1.3)
    grid1.yaxis.grid(which = "both")

    # plot!
    plt.savefig(home+'P(k)_comparison_'+color+'_'+hodtype+'_'+density+'_'+str(res))
    plt.clf()    

def covariance_sanity_check(matter_k, pm, gal_k, kvals_file, lin_pk_file, res, nmesh):
    
    # do it with chun-hao's field
    print('covariance debugging:', np.shape(matter_k), np.shape(gal_k))
    lin_pk_gal_matter, kvals_gal_matter = pk(matter_k, 'gal_k_matter_k', kvals_file, lin_pk_file, res, correlation = 'cross', 
                                             second = gal_k)
    lin_pk_matter, kvals_matter = pk(matter_k, 'matter_k', kvals_file, lin_pk_file, res)
    
    np.save(lin_pk_file+'_biasval2_2', lin_pk_gal_matter/lin_pk_matter - 1) # sanity check for fig 10!
    
    # now do it with nick's field
    matter_field_file = '/scratch/users/kokron/tngsnaps/snap_50/tng_deltam.npy'
    if os.path.isfile(matter_field_file):
        matter_field = np.load(matter_field_file)
        mpiprint('matter: '+str(np.shape(matter_field)))

        matter_k, narr = fieldread_mpi_original(matter_field, pm, nmesh)
        matter_k = matter_k.r2c()   

        print('covariance debugging:', np.shape(matter_k), np.shape(gal_k))
        lin_pk_gal_matter, kvals_gal_matter = pk(matter_k, 'gal_k_matter_k', kvals_file, lin_pk_file, res, correlation = 'cross', 
                                                 second = gal_k)
        lin_pk_matter, kvals_matter = pk(matter_k, 'matter_k', kvals_file, lin_pk_file, res)
        np.save(lin_pk_file+'_biasval_2', lin_pk_gal_matter/lin_pk_matter - 1) 

def run_perr(bigbvec, perrvec, snapshot, pm, box, Lbox, color, hodtype, density, nmesh, res, Ndown, D, nbias, kmaxvec, kmaxfin, z,
             comp = True):
   
    hodslice, nbar = halocat_to_cut(snapshot, Lbox, color, hodtype, density, z)
    mpiprint(np.shape(hodslice))
    
    nbarmean = MPI_mean(nbar)
    mpiprint('Mean MPI nbar is %.6f'%nbarmean) 
    
    layout = pm.decompose(hodslice)
    p = layout.exchange(hodslice)
    mpiprint('p: '+str(np.shape(p)))
    gal_field = pm.create(type='real')
    mpiprint('gal_field: '+str(np.shape(gal_field)))
    pm.paint(p, out=gal_field, mass=1, resampler='cic')
    
    # Make overdensity contrast
    gal_field = gal_field / gal_field.cmean() - 1
    mpiprint(gal_field.shape)
    
    gal_k = gal_field.r2c()    
    if res == 128: 
        mpiprint('res: '+str(res))
        kvals_file = home+'kvals_'+color+'_'+hodtype+'_'+density+'_128'
        lin_pk_file = home+'lin_pk_'+color+'_'+hodtype+'_'+density+'_128'
        if os.path.isfile(kvals_file+'.npy') & os.path.isfile(lin_pk_file+'.npy'):
            print(kvals_file)
            sanity_check(gal_k, None, Lbox, nbar, color, hodtype, density, kvals_file, lin_pk_file, res) 
        print('debugging exit\n')
        exit()

    del gal_field
    gc.collect()
    mpiprint(time.time() - st) 
    sys.stdout.flush()
    
    ####Load in fields and convert to Fourier-space    
    if z == 0 or z == 1:
        boxdir = '/oak/stanford/orgs/kipac/users/kokron/anzu_tng/lagfiles/'
    else:
        boxdir = '/oak/stanford/orgs/kipac/users/mahlet/TNG300-1/lagfield/'
    density_field = np.load(boxdir+'latetime_weight_0_5000_'+str(res)+'_z'+str(z)+'_rank'+str(rank)+'.npy', mmap_mode='r') 
    print('density', np.shape(density_field), 'rank:', rank)
    density_k, narr = fieldread_mpi(density_field, pm, nmesh)
    density_k = density_k.r2c()
    mpiprint('Loaded density!')
    
    del density_field
    gc.collect()
    sys.stdout.flush()
    
    lin = np.load(boxdir+'latetime_weight_1_5000_'+str(res)+'_z'+str(z)+'_rank'+str(rank)+'.npy', mmap_mode='r') 
    lin_k, _ = fieldread_mpi(lin, pm, nmesh)
    lin_k = lin_k.r2c()*D
    del lin
    
    gc.collect()
    mpiprint('Loaded lin!')
    
    sys.stdout.flush()
    quad = np.load(boxdir+'latetime_weight_2_5000_'+str(res)+'_z'+str(z)+'_rank'+str(rank)+'.npy', mmap_mode='r') 
    quad_k, _ = fieldread_mpi(quad, pm, nmesh) 
    quad_k = quad_k.r2c()*D**2
    del quad
    
    gc.collect()
    mpiprint('Loaded quad!')
    
    sys.stdout.flush()
    tide = np.load(boxdir+'latetime_weight_3_5000_'+str(res)+'_z'+str(z)+'_rank'+str(rank)+'.npy', mmap_mode='r') 
    tide_k, _ = fieldread_mpi(tide, pm, nmesh)
    tide_k = tide_k.r2c()*D**2
    del tide
    
    gc.collect()
    nabla2 = np.load(boxdir+'latetime_weight_4_5000_'+str(res)+'_z'+str(z)+'_rank'+str(rank)+'.npy', mmap_mode='r') 
    nabla_k,_= fieldread_mpi(nabla2, pm, nmesh)
    nabla_k = nabla_k.r2c()*D
    del nabla2
    
    gc.collect()
    cubic = np.load(boxdir+'latetime_weight_5_5000_'+str(res)+'_z'+str(z)+'_rank'+str(rank)+'.npy', mmap_mode='r') 
    cubic_k,_= fieldread_mpi(cubic, pm, nmesh)
    cubic_k = cubic_k.r2c()*D**3
    del cubic
    
    gc.collect()    
    mpiprint('test: %d'%(time.time() - st)) 
    sys.stdout.flush()
    bias_fieldlist = [lin_k, quad_k, tide_k, nabla_k, cubic_k]
        
    gal_denssub = gal_k - density_k
    
    gc.collect()
    
    crosscorrmat = np.zeros(shape=(nbias, nbias), dtype='complex64')
    diffarray = np.zeros(shape=(nbias), dtype='complex64')
    
    knorm = fieldknorm(gal_denssub) 
    for n, kmax in enumerate(kmaxvec):
        kidx = knorm < kmax
        for i in range(nbias):
            
            # subselect on the modes that are |k| < kmax 
            gsub = gal_denssub[kidx]  
            
            gsub = np.conj(gsub)

            fsub = bias_fieldlist[i][kidx]
            
            # This is the local sum for this rank. Sum across all. 
            Asum = np.sum(gsub*fsub)

            allsum = MPI_sum(Asum)
            diffarray[i] = allsum 
            for j in range(i+1):
        
                fsub2 = bias_fieldlist[j][kidx]
                fsub2 = np.conj(fsub2)

                msum = np.sum(fsub*fsub2)
                crosscorrmat[i,j] = MPI_sum(msum) 
                crosscorrmat[j,i] = 1.*crosscorrmat[i,j]
            sys.stdout.flush()
        invmats = np.linalg.inv(crosscorrmat.T).T
        bvals = np.einsum('ij, j->i', invmats, diffarray).real
        
        bigbvec[n] = bvals  
        
        invmatvec.append(invmats) 
    mpiprint(time.time() - st)
    k1_idx = find_nearest(kmaxvec, kmaxfin)
    np.save(home+'k1_idx', k1_idx)
    mpiprint('kmax: '+str(kmaxvec[k1_idx]))
    bvals = bigbvec[k1_idx] 
    biasfield = 0
    for i in range(nbias):
            biasfield+= bvals[i] * bias_fieldlist[i]
    
            epsilon_k = gal_denssub - biasfield
    
    epsilon_k = gal_denssub - biasfield
    
    # sanity check the bias params against the estimate using covariance
    kvals_file = home+'kvals_'+color+'_'+hodtype+'_'+density+'_'+str(res)+'_z'+str(z) 
    lin_pk_file = home+'lin_pk_'+color+'_'+hodtype+'_'+density+'_'+str(res)+'_z'+str(z)
    covariance_sanity_check(density_k, pm, gal_k, kvals_file, lin_pk_file, res, nmesh)
    
    # sanity check the HEFT clustering model!
    mpiprint('Commencing... '+str(os.path.isfile(kvals_file+'.npy') & os.path.isfile(lin_pk_file+'.npy')))
    if os.path.isfile(kvals_file+'.npy') & os.path.isfile(lin_pk_file+'.npy'):
        mpiprint('Sanity Check!')
        sanity_check(gal_k, density_k + biasfield, Lbox, nbar, color, hodtype, density, kvals_file, lin_pk_file, res) 
    else:
        mpiprint('No sanity check!')
    del gal_k, density_k
    
    if comp:
        epsilon_k = epsilon_k.apply(CompensateCICAliasing, kind='circular')
    perr = FFTPower(epsilon_k, '1d', kmin = 1e-5)
    
    mpiprint('perrvec: '+ str(np.shape(perrvec)))
    mpiprint('perr: '+str(np.shape(perr.power['power'].real)))
    perrvec = perr.power['power'].real
    kvals = perr.power['k']
    mpiprint('Test box done in %.2f :'%(time.time() - st))
    del epsilon_k, bias_fieldlist, gal_denssub, lin_k, quad_k, tide_k
    gc.collect()

    return perrvec, bigbvec, invmatvec, kvals, nbarmean

def main():
     
    mpiprint('nranks: %d'%(nranks))
    box     = 6
    bigseed = np.random.randint(1e5)
    mpiprint('Bigseed is given by %d'%bigseed)

    hodtype = sys.argv[1] 

    # final kmax since we will run estimator for many
    kmaxfin = float(sys.argv[3])
    kmaxvec = np.linspace(0.05, 1.0, 40)
    nbias = int(sys.argv[4]) 
    snapshot = int(sys.argv[2])
    zvec = {99: 0,
            67: 0.5,
            50: 1,
            40: 1.5}
    z = zvec[snapshot]

    Ndown = 1
    res = int(sys.argv[5]) 
    nmesh = res//Ndown 
    bigbvec = np.zeros(shape=(len(kmaxvec), nbias))
    if nmesh%2==0:
        perrvec = np.zeros(shape=(nmesh//2))
    else:
        perrvec = np.zeros(shape=(nmesh//2+1)) 
    Lbox = 205
    R=0.75

    cosmofiles = pd.read_csv('/home/users/kokron/Projects/lakelag/test_cosmos.txt', sep=' ')
    box0cosmo = cosmofiles.iloc[-1]
    cosmo = ccl.Cosmology(Omega_b = box0cosmo['ombh2']/(box0cosmo['H0']/100)**2, Omega_c = box0cosmo['omch2']/(box0cosmo['H0']/100)**2,
                           h = box0cosmo['H0']/100, n_s = box0cosmo['ns'], w0=box0cosmo['w0'], Neff=box0cosmo['Neff'],
                            sigma8 = box0cosmo['sigma8'])
    k = np.logspace(-5, 2.3, 1000)

    # Computing growth factor
    boxcosmo = {'ombh2': 0.02230,
                'H0': 67.74,
                'omch2': 0.11944,
                'ns': 0.9667,
                'w0': -1.0,
                'Neff': 3.046,
                'sigma8': 0.8159}
    configs = {'z_ic': 127}
    cosmo = ccl.Cosmology(Omega_b = boxcosmo['ombh2']/(boxcosmo['H0']/100)**2, Omega_c = boxcosmo['omch2']/(boxcosmo['H0']/100)**2, 
                          h = boxcosmo['H0']/100, n_s = boxcosmo['ns'], w0 = boxcosmo['w0'], Neff = boxcosmo['Neff'],sigma8 =
                          boxcosmo['sigma8'])

    # Aemulus boxes have IC at z=49
    z_ic=configs['z_ic']
    
    # Compute relative growth from IC to snapdir 
    box_scale = 1/(1+z)
    mpiprint('a: %f'%(box_scale))
    growthratio = ccl.growth_factor(cosmo, [box_scale])/ccl.growth_factor(cosmo, 1./(1+z_ic))
    
    # Vector to rescale component spectra with appropriate linear growth factors.
    D = growthratio
    pm = pmesh.pm.ParticleMesh([nmesh, nmesh, nmesh], Lbox, dtype='float32', resampler='cic', comm=comm)
   
    ### Start here
    color, density = sys.argv[6:] 
    sys.stdout.flush()
    perrvec, bigbvec, invmatvec, kvals, nbarmean = run_perr(bigbvec, perrvec, snapshot, pm, box, Lbox, color, hodtype, density, nmesh,
                                                            res, Ndown, D, nbias, kmaxvec, kmaxfin, z)
    gc.collect()
    sys.stdout.flush()
    time.sleep(10)
                            
    if rank==0:
        np.save('mpiksumperr_%shod_z%s_kmax%.2f_Ndown%d_nbias%d_%s_%s'%(hodtype,z,kmaxfin,Ndown,nbias, color, density), perrvec)
        np.save('mpibiasval_%shod_z%s_Ndown%d_nbias%d_%s_%s'%(hodtype,z,Ndown,nbias, color, density), bigbvec)
        np.save('mpiinvmat_%shod_z%s_Ndown%d_nbias%d_%s_%s'%(hodtype,z,Ndown,nbias, color, density), np.array(invmatvec))
        np.save('mpikmax_%shod_z%s_Ndown%d_nbias%d_%s_%s'%(hodtype,z,Ndown,nbias, color, density), np.array(kmaxvec))
        np.save('mpikvals_%shod_z%s_Ndown%d_nbias%d_%s_%s'%(hodtype,z,Ndown,nbias, color, density), np.array(kvals))
        np.save('mpinbar_%shod_z%s_Ndown%d_nbias%d_%s_%s'%(hodtype,z,Ndown,nbias, color, density), nbarmean)
    
    mpiprint('Success! '+hodtype+color+density)
    
if __name__ == "__main__":
    main()
