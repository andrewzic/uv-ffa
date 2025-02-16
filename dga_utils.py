#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from casacore.tables import *
from argparse import ArgumentParser
from tqdm import tqdm
import time
#from numba import njit
import numba as nb
import sys

#import dask
import dask.array as da
#from dask_jobqueue import SlurmCluster

def get_nbaselines(n_antennas=30):
    return int(n_antennas * (n_antennas - 1)/2)

def get_data_and_flags(t):
    vis_data_all = t.getcol('DATA')
    vis_flag_all = t.getcol('FLAG')
    return vis_data_all, vis_flag_all

def get_dask_data_and_flags(t):
    vis_data_all = da.array(t.getcol('DATA'))
    vis_flag_all = da.array(t.getcol('FLAG'))
    return vis_data_all, vis_flag_all

def get_channel_lambdas(ms):
    tf = table(f"{ms}/SPECTRAL_WINDOW")
    # get all channel frequencies and convert that to wavelengths
    channel_freqs = tf[0]["CHAN_FREQ"]
    nchan = len(channel_freqs)
    channel_lambdas = 299792458 / channel_freqs
    tf.close()
    return nchan, channel_freqs, channel_lambdas


def get_time(t):
    vis_time = t.getcol('TIME_CENTROID')
    unique_times = np.unique(vis_time)
    assert np.all(np.abs(np.diff(unique_times) - np.diff(unique_times)[0]) < 1e-2)
    nsub = unique_times.shape[0]
    return nsub, vis_time, unique_times



def get_uvwave_indices(t, channel_lambdas, uvcell_size, N_pixels):
    
    # now we want to get the uv positions (in lambdas) for each visibility sample
    uvws = t.getcol('UVW') # in units of metres

    uvws_l = np.ones((uvws.shape[0], uvws.shape[1], len(channel_lambdas))) * uvws[:, :, None]
    # duplicate each uv sample into a size such that we can multiply with the channel lambdas    
    uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
    
    chan_tiled = np.ones_like(uvs_l)*channel_lambdas[None, None, :] # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
    
    uvs_l = uvs_l / chan_tiled    # get the uvw positions in terms of the number of lambda instead of metres

    print(f"max uvwave is {np.max(np.sqrt(uvs_l[:, 0, :]**2.0 + uvs_l[:, 1, :]**2.0)):.3f}")
    
    # now get the uv grid indices for each uv sample
    uv_indices = np.round(uvs_l/uvcell_size).astype(int) + N_pixels//2
    u_indices = uv_indices[:, 0, :]
    v_indices = uv_indices[:, 1, :]
    uv_indices_compl = (u_indices + 1j*v_indices).flatten()
    unique_uv_indices = np.unique(uv_indices_compl)
    nuvcell = int(unique_uv_indices.shape[0])
    return nuvcell, uvs_l, uv_indices, unique_uv_indices

#@dask.delayed
def get_uvwave_indices_dask(t, channel_lambdas, uvcell_size, N_pixels):
    
    # now we want to get the uv positions (in lambdas) for each visibility sample
    uvws = da.array(t.getcol('UVW')) # in units of metres

    uvws_l = da.ones((uvws.shape[0], uvws.shape[1], len(channel_lambdas))) * uvws[:, :, None]
    # duplicate each uv sample into a size such that we can multiply with the channel lambdas    
    uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
    
    chan_tiled = da.ones_like(uvs_l)*channel_lambdas[None, None, :] # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
    
    uvs_l = uvs_l / chan_tiled    # get the uvw positions in terms of the number of lambda instead of metres

    print(f"max uvwave is {np.max(np.sqrt(uvs_l[:, 0, :]**2.0 + uvs_l[:, 1, :]**2.0)):.3f}")
    
    # now get the uv grid indices for each uv sample
    uv_indices = np.round(uvs_l/uvcell_size).astype(int) + N_pixels//2
    u_indices = uv_indices[:, 0, :]
    v_indices = uv_indices[:, 1, :]
    uv_indices_compl = (u_indices + 1j*v_indices).flatten()
    unique_uv_indices = np.unique(uv_indices_compl)
    nuvcell = int(unique_uv_indices.shape[0])
    return nuvcell, uvs_l, uv_indices, unique_uv_indices


def get_uv_grid():
    #FoV_diameter = 2.4 #deg
    #FoV_diameter *= np.pi/180.0 #convert to radians
    #max_uv = 8500 #hard-coded... should check what the absolute max baseline length is for askap antennas < 30
    #psf_diameter = 1.0/max_uv #this will be in radians
    #image_pixel_size = psf_diameter / 2.1 #also in radians. we just-oversample the psf
    #this is 754 or so which is close to...
    N_pixels = 768 ##2**8 * 3
    max_uv = 8500.0
    uvcell_size = 2.0*max_uv / N_pixels
    uv_bins = np.arange(-max_uv, max_uv, uvcell_size)
    uv_grid = np.zeros((N_pixels, N_pixels))
    return N_pixels, max_uv, uvcell_size, uv_bins, uv_grid


def get_uv_bins_old():
    N_pix = 384
    N_pixels = 384  # number of pixels for the uv grid and will eventually be (N_pixels/2)xN_pixels for the dynamic grid precursor
    max_uv = 8700.0 #hard coding for now
    #np.max(np.abs(uvs_l)) # our grid needs to extend to the positive and negative max_uv value on both axes
    uv_grid = np.zeros((N_pixels, N_pixels))
    bins = np.linspace(-max_uv, max_uv, N_pixels)
    uvcell_size = bins[1] - bins[0]


def amend_flags(flags, rowinds):
    flags[rowinds] = True
    return(flags)

def zeroflag_ms(vis_data, flags):
    
    #flags = t.getcol('FLAG')
    #visibility_data = (t.getcol('DATA'))

    #zero-ing out flagged data for now
    vis_data[flags == True] = 0.0 + 1j*0.0
    
    return(vis_data)

#@dask.delayed
def zeroflag_ms_dask(vis_data, flags):
    
    #flags = t.getcol('FLAG')
    #visibility_data = (t.getcol('DATA'))

    #zero-ing out flagged data for now
    vis_data[flags == True] = 0.0 + 1j*0.0
    
    return(vis_data)

"""
def zeroflag_rows(vis_data, flagrow_inds):
    vis_data[flagrow_inds, :] = 0.0 + 1j*0.0
    
    return(vis_data)
"""
#@dask.delayed
def zeroflag_rows(vis_data, flagrow_inds):
    vis_data[flagrow_inds, :] = 0.0 + 1j*0.0
    
    return(vis_data)



def flag_askap_long_baselines(t):
    antenna2 = t.getcol('ANTENNA2')
    use_baselines = np.argwhere(antenna2 < 29).squeeze()

    t = t.selectrows(use_baselines)
    return t

def grab_first_100_ints(t):
    time = t.getcol('TIME_CENTROID')
    unique_times = np.unique(time)
    use_times = np.argwhere(time < unique_times[10]).squeeze()
    t = t.selectrows(use_times)
    return t

def flag_flaggedrows(t, vis_data_all, vis_flag_all):
    flagged = t.getcol('FLAG_ROW')
    flagged = flagged.astype(bool)
    flagrow_inds = np.argwhere(flagged).squeeze()
    
    not_flagged = ~flagged
    rownrs = np.arange(len(not_flagged))
    #t = t.selectrows(rownrs[not_flagged])
    
    vis_data_all = zeroflag_rows(vis_data_all, flagrow_inds)
    vis_flag_all = amend_flags(vis_flag_all, flagrow_inds)
    return vis_data_all, vis_flag_all

def flag_zerouvw(t, vis_data_all, vis_flag_all):
    uvws = t.getcol('UVW') # in units of metres
    uvw0_inds = np.argwhere(np.abs(np.sum(uvws, axis=1)) < 0.1).squeeze()
    #uvw0_inds = np.argwhere(np.abs(np.sum(uvws, axis=1)) > 0.1).squeeze()
    #t = t.selectrows(inds)
    vis_data_all = zeroflag_rows(vis_data_all, uvw0_inds)
    vis_flag_all = amend_flags(vis_flag_all, uvw0_inds)
    return vis_data_all, vis_flag_all

#@njit
def make_stokesI(vis_data):
    visibilities_I = ( vis_data[:, :, 0] + vis_data[:, :, -1])/2.0
    return(visibilities_I)

def make_weights(t):
    flags = t.getcol('FLAG')
    weights = (flags == False).astype(int)
    return weights

#@njit
def make_weights_from_flags(flags):
    weights = (flags == False).astype(int)
    return weights


print("working out unique uv indices over all times channels etc")
#@dask.delayed
def get_unique_uv_indices_per_timestep(uv_indices):
    uv_inds_unique = []
    for uvi_ind in range(uv_indices.shape[0]):
        uv_inds_unique.append(np.unique(uv_indices[uvi_ind, :, :], return_inverse = False))

    uv_inds_unique = np.array(uv_inds_unique, dtype = 'object')
    return uv_inds_unique

def clean_uv_indices_per_timestep(unique_uv_indices_timestep, N_pixels):
    #stub to be written
    pass
    #for

def organise_data_cubes(nsub, nbl, nchan, npol, vis_data_I, uv_indices_all, vis_weights_all, vis_flag_all):
    #now reshape into nbl x nsub x nchan array
    vis_data = vis_data_I.reshape(nsub, nbl, nchan)#, vis_data_all.shape[2])
    #uv_indices into nbl x nsub x 2 x nchan array
    uv_indices =uv_indices_all.reshape(nsub, nbl, uv_indices_all.shape[1], nchan)
    #form uv indices into a complex int for compact representation and portability
    uv_indices = uv_indices[:, :, 0, :] + 1j * uv_indices[: ,: , 1, :]
    #reshape weights
    vis_weights = vis_weights_all.reshape(nsub, nbl, nchan, npol)
    #reshape flags too
    vis_flag = vis_flag_all.reshape(nsub, nbl, nchan, npol)
    print("organised data into nice shapes")

    return vis_data, uv_indices, vis_weights, vis_flag

#@dask.delayed
def organise_data_cubes_dask(nsub, nbl, nchan, npol, vis_data_I, uv_indices_all, vis_weights_all, vis_flag_all):
    #now reshape into nbl x nsub x nchan array
    vis_data = vis_data_I.reshape(nsub, nbl, nchan)#, vis_data_all.shape[2])
    #uv_indices into nbl x nsub x 2 x nchan array
    uv_indices =uv_indices_all.reshape(nsub, nbl, uv_indices_all.shape[1], nchan)
    #form uv indices into a complex int for compact representation and portability
    uv_indices = uv_indices[:, :, 0, :] + 1j * uv_indices[: ,: , 1, :]
    #reshape weights
    vis_weights = vis_weights_all.reshape(nsub, nbl, nchan, npol)
    #reshape flags too
    vis_flag = vis_flag_all.reshape(nsub, nbl, nchan, npol)
    print("organised data into nice shapes")

    return vis_data, uv_indices, vis_weights, vis_flag


#@njit
def make_dga(vis_data_all, uv_indices_all, vis_weights_all, unique_times, vis_time):

    """
    construct a dense DGA in a stupid un-pythonic triple nested for-loop
    this could be implemented in C or julia much faster...
    """
    
    #uv_ind_array = np.array([],dtype='object')
    #weights_array = np.array([],dtype='object')
    #dga = np.array([],dtype='object')
    
    uv_ind_array = []
    weights_array = []
    dga = []

    for it in range(len(unique_times)):
        if it % 10 == 0:
            print(it / len(unique_times))
        #if it == len(unique_times)*0.75
        uv_list = np.array([])
        weights_list = np.array([])
        dga_list = np.array([])
        
        # looping over each unique time, and populating the uv grid for that time with visibilities:
        # we need to find the indices that correspond to the current time
        #assuming all times across baselines are the same
        valid_times = vis_time == unique_times[it]
        time_it = unique_times[it]
        #t2 = taql("select * from $t where TIME_CENTROID == $time_it")
        
        nbl = np.sum(valid_times)
        
        #uvs_l, uv_indices = get_uvwave_indices(t2, channel_lambdas, uvcell_size, N_pixels)
        #u_indices = uv_indices[:, 0, :]
        #v_indices = uv_indices[:, 1, :]
        
        #uvs_l = uvs_l_all[valid_times]
        uv_indices=uv_indices_all[valid_times]
        
        #visibility_data = t2.getcol('DATA')
        visibility_data = vis_data_all[valid_times]
        
        #weights = make_weights(t2)
        weights = vis_weights_all[valid_times]
        
        #visibility_data = zeroflag_ms(t2, visibility_data)
        visibilities_I = make_stokesI(visibility_data)
        weights_I = make_stokesI(weights) #samples with one pol flagged will == 0.5
        
        for ibl in range(nbl):
            for ich in range(nchan):
                
                uv_samp = (uv_indices[ibl,:,  ich].squeeze())
                #uv_samp = (uv_indices[ibl,:,  ich][0][0])#.squeeze())
                
                #for convenience of having a 1D list we cast uv_samp to a complex number:
                uv_samp = uv_samp[0] + 1j*uv_samp[1]
                vis_samp = visibilities_I[ibl, ich]
                weights_samp = weights_I[ibl, ich]
                
                if np.any(np.isin(uv_list, uv_samp)):
                    ind = np.argwhere(np.isin(uv_samp, uv_list)).squeeze() #should only be one
                    if ind.size > 1:
                        raise ValueError('More than one element of the uv list contains the current uv sample- this shouldnt happen')
                    if ind.size == 0:
                        raise ValueError('No element of the uv list contains the current uv sample- this shouldnt happen')
                    
                    ind = ind.item() #convert 0-dimensional array to number
                    weights_list[ind] += weights_I[ibl, ich]
                    dga_list[ind] += visibilities_I[ibl, ich]
                    
                    # if (-u, -v) is the current coord and (u, v) is already sampled, then we conjugate and add to (u, v)
                elif np.any(np.isin(uv_list, (N_pixels + 1j*N_pixels -1*uv_samp))):
                    ind = np.argwhere(np.isin(uv_list, (N_pixels + 1j*N_pixels -1*uv_samp))).squeeze()
                    if ind.size > 1:
                        raise ValueError('More than one element of the uv list contains the current uv sample- this shouldnt happen')
                    if ind.size == 0:
                        raise ValueError('No element of the uv list contains the current uv sample- this shouldnt happen')
                    
                    ind = ind.item()
                    
                    weights_list[ind] += weights_I[ibl, ich]
                    dga_list[ind] += np.conj(visibilities_I[ibl, ich]) #conjugate and add to (u,v) ind
                else:
                    uv_list = np.append(uv_list, uv_samp)
                    weights_list = np.append(weights_list, weights_I[ibl, ich])
                    dga_list = np.append(dga_list, visibilities_I[ibl, ich])
                    
                
            #uv_ind_array = np.append(uv_ind_array, uv_list)
            uv_ind_array.append(uv_list)
            #weights_array = np.append(weights_array, weights_list)
            weights_array.append(weights_list)
            #dga = np.append(dga, dga_list)
            dga.append(dga_list)
            #t2.close()
        return dga, uv_ind_array, weights_array


def find_sum_uv_vis(uv_indices_it, vis_data_it, uv_ind_unique, uv_inds_dense_it, N_pixels, dga_dense_it):
    """
    for a single time iteration and a given (u, v) cell, find all visibility samples overlapping in this (u,v) cell and sum up
    and also find any (-u, -v) samples and add their conjugate sum 
    """
    
    uv_inds_ = uv_indices_it == uv_ind_unique
    conj_inds_ = uv_inds_dense_it == N_pixels*(1+1j) - uv_ind_unique
    
    # if the -u, -v is already in the array, add to there
    if np.any(conj_inds_):
        dga_dense_it[conj_inds_] = dga_dense_it[conj_inds_] + np.sum(np.conj(vis_data_it[uv_inds_]))
    else:
        dga_dense_it = np.append(dga_dense_it, np.sum(vis_data_it[uv_inds_]))
        uv_inds_dense_it = np.append(uv_inds_dense_it, uv_ind_unique)
        
    return uv_inds_dense_it, dga_dense_it

def sum_uv_vis_grid_it(uv_inds_unique_it, uv_indices_it, vis_data_it, N_pixels):
    """
    nested function to run in a single time iteration
    goes through all uv cells sampled at time[it] and sums up visibilities accordingly
    """
    
    dga_dense_it = np.array([])
    uv_inds_dense_it = np.array([])
            
    for uv_ind_unique in uv_inds_unique_it:
        #inds = np.array([np.isin(uv_ind_array_j, uvi) for uv_ind_array_j in uv_ind_array])
        uv_inds_dense_it, dga_dense_it = find_sum_uv_vis(uv_indices_it, vis_data_it, uv_ind_unique, uv_inds_dense_it, N_pixels, dga_dense_it)
        
    return uv_inds_dense_it, dga_dense_it
        

#@dask.delayed
def sum_uv_vis_grid(nsub, uv_inds_unique, uv_indices, vis_data, N_pixels, dga_dense, uv_inds_dense):
    """ goes through each nsub time integration and appends list of gridded summed visibilities and uv cells"""
    
    for it in range(nsub):
        uv_inds_dense_it, dga_dense_it = sum_uv_vis_grid_it(uv_inds_unique[it], uv_indices[it, :, :], vis_data[it, :, :], N_pixels)
        dga_dense.append(dga_dense_it)
        uv_inds_dense.append(uv_inds_dense_it)
        #uv_ind_array = np.append(uv_ind_array, uv_list)

    return uv_inds_dense, dga_dense


def make_dga_grid(nsub, nchan, nbl, vis_data, uv_indices, vis_weights, unique_times, vis_time, uv_inds_unique, N_pixels):
        #uv_ind_array = np.array([],dtype='object')
        #weights_array = np.array([],dtype='object')
        #dga = np.array([],dtype='object')

        uv_ind_array = []
        weights_array = []
        dga = []

        #dga_dense = np.array([], dtype='object')
        #uv_inds_dense = np.array([], dtype='object')
        dga_dense = []
        uv_inds_dense = []
        #dga_dense = np.array([np.zeros((i.shape[0])) for i in uv_inds_unique], dtype='object')
        #uv_inds_unique_inds = np.array([np.unique_index(uv_indices[i, :, :]) for i in range(uv_indices.shape[1])], dtype='object')

        uv_inds_dense, dga_dense = sum_uv_vis_grid(nsub, uv_inds_unique, uv_indices, vis_data, N_pixels, dga_dense, uv_inds_dense)

        #dga_dense = np.array(dga_dense, dtype='object')
        #uv_inds_dense = np.array(uv_inds_dense, dtype='object')
        return dga_dense, uv_inds_dense


def zeropad_dense_dga(nuvcell, nsub, dga_dense, uv_ind_array, unique_uv_indices, unique_times ):
    
    dga_all = np.zeros((nuvcell+1, nsub+1), dtype=np.complex128)
    #for each uv index in all of the uv-indices touched by the array
    #set the first column, after the first row, to the (complex) uv-index
    dga_all[1:, 0] = unique_uv_indices
    #set the first row, after the first col, to the time sample    
    dga_all[0, 1:] = unique_times
    
    for i, uvi in tqdm(enumerate(unique_uv_indices)):
    
        # find times where this uv sample touches
        #note reverse order of isin(array, sample) returns boolean array of same shape of input array
        inds = np.array([np.isin(uv_ind_array_j, uvi) for uv_ind_array_j in uv_ind_array], dtype = 'object')
        #time samples corresponding to the uv cell
        colinds = np.argwhere([ind.any() for ind in inds]).squeeze()
        
        if colinds.size < 0.1 * nsub:
            #ignore uv cells that have >90% empty time series
            continue
        
        #which element from this time sample are we grabbing
        if colinds.size > 1:

            rowinds = [np.argwhere(ind).squeeze() for ind in inds[colinds]]
            
            for j, rowind in zip(colinds, rowinds):
                dga_all[i+1, j+1] = dga_dense[j][rowind]
                
        elif colinds.size == 1:
            rowinds = np.array([np.argwhere(inds[colinds]).item()])
            dga_all[i+1, colinds+1] = dga_dense[colinds][rowinds]

    return dga_all


def main():

    ap = ArgumentParser(description="Get frequency and time brightness data from interferometric visibilities.")
    ap.add_argument('-ms', '--msname', type=str, help='Path to measurement set file for the complete observation.')
    args = ap.parse_args()
    ms = args.msname #"/scratch3/zic006/ffa/ffa/scienceData.J183950.5-075635.SB58609.J183950.5-075635.beam13_averaged_cal.leakage.uvsub.ms"

    nchan, channel_freqs, channel_lambdas = get_channel_lambdas(ms)

    t = table(ms, readonly=False, memorytable=True)
    
    #count number of baselines
    nbl = get_nbaselines(30)

    #get the uv grid properties
    N_pixels, max_uv, uvcell_size, bins, uv_grid = get_uv_grid()
    
    # now get rid of data on the longest baselines (antenna2 >= 30 are removed)
    # we get rid of those large baselines as they are sparsely sampled in uv space and so just adds unnecessary computational cost
    t = flag_askap_long_baselines(t)

    #let's make sure we're being sensible here...
    t = t.sort('TIME_CENTROID')

    t = grab_first_100_ints(t)

    vis_data_all, vis_flag_all = get_data_and_flags(t)
    npol = vis_data_all.shape[2]

    # get rid of flagged time samples
    vis_data_all, vis_flag_all = flag_flaggedrows(t, vis_data_all, vis_flag_all)
    
    # now get rid of data with a very low or 0 uv-distance
    vis_data_all, vis_flag_all = flag_zerouvw(t, vis_data_all, vis_flag_all)

    # tie a bow on top of everything
    vis_data_all = zeroflag_ms(vis_data_all, vis_flag_all)
    
    print("Done flagging data")

    # get time info
    nsub, vis_time, unique_times = get_time(t)

    nuvcell, uvs_l_all, uv_indices_all, unique_uv_indices = get_uvwave_indices(t, channel_lambdas, uvcell_size, N_pixels)
    t.close()
    
    print("grabbed uv indices")
    
    #make the weights from the flags - set flagged data points to have zero weight
    vis_weights_all = make_weights_from_flags(vis_flag_all)
    
    #sum up XX and YY and corresponding weights to form total-intensity visibilities
    vis_data_I = make_stokesI(vis_data_all)
    vis_weights_I = make_stokesI(vis_weights_all)
    print("made Stokes I")
    
    
    #now need to return useful stuff. also need to define npol etc. in main pipelin
    vis_data, uv_indices, vis_weights, vis_flag = organise_data_cubes(nsub, nbl, nchan, npol, vis_data_I, uv_indices_all, vis_weights_all, vis_flag_all)

    #work out unique indices
    unique_uv_indices_timestep = get_unique_uv_indices_per_timestep(uv_indices)
    #and then correct them for conjugation over time
    # not implemented yet
    #unique_uv_indices_timestep = clean_uv_indices_per_timestep(unique_uv_indices_timestep, N_pixels)


    print("Forming uv-grid cells through time...")

    #now make the dense dga and store the uv indices
    dga_dense, uv_ind_array = make_dga_grid(nsub, nchan, nbl, vis_data, uv_indices, vis_weights, unique_times, vis_time, unique_uv_indices_timestep, N_pixels)

    print("done")
    #dga, uv_ind_array, weights_array = make_dga(vis_data_all, uv_indices_all, vis_weights_all, unique_times, vis_time)
    #now need to unpack dga into usable format

    #set up zero-padded DGA
    print("setting up zero-padded DGA")
    dga_all = zeropad_dense_dga(nuvcell, nsub, dga_dense, uv_ind_array, unique_uv_indices, unique_times)

    #save the output
    print("saving output")
    np.save(f'{ms.replace(".ms", "")}.dga_dense.npy', dga_dense)
    np.save(f'{ms.replace(".ms", "")}.times.npy', unique_times)
    print("saved dense DGA")
    np.save(f'{ms.replace(".ms", "")}.uv_indices.npy', unique_uv_indices)
    np.save(f'{ms.replace(".ms", "")}.dga.npy', dga_all)

if __name__ == "__main__":
    main()
