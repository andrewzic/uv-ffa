#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from casacore.tables import *
from argparse import ArgumentParser
from tqdm import tqdm
import time


ap = ArgumentParser(description="Get frequency and time brightness data from interferometric visibilities.")
ap.add_argument('-ms', '--msname', type=str, help='Path to measurement set file for the complete observation.')

args = ap.parse_args()

ms = args.msname

t = table(ms, readonly=False, memorytable=True)




time1 = time.time()

# get rid of flagged time samples
flagged = t.getcol('FLAG_ROW')
flagged = flagged.astype(bool)
not_flagged = ~flagged
rownrs = np.arange(len(not_flagged))
t = t.selectrows(rownrs[not_flagged])

# now get rid of data with a very low uv-distance
uvws = t.getcol('UVW') # in units of metres
inds = np.argwhere(np.abs(np.sum(uvws, axis=1)) > 0.1).squeeze()
t = t.selectrows(inds)

# now get rid of data on the longest baselines (antenna2 >= 30 are removed)
# we get rid of those large baselines as they are sparsely sampled in uv space and so just adds unnecessary computational cost for baselines above ~9k lambda
antenna2 = t.getcol('ANTENNA2')
rownrs = np.arange(len(antenna2))
use_baselines = np.where(antenna2 < 30)
t = t.selectrows(rownrs[use_baselines])

tf = table(f"{ms}/SPECTRAL_WINDOW")
vis_time = t.getcol('TIME')
vis_feed = t.getcol('FEED1')
vis_scan = t.getcol('SCAN_NUMBER')

# get each unique timestamp
unique_times = np.unique(vis_time)

# get all channel frequencies and convert that to wavelengths
channel_freqs = tf[0]["CHAN_FREQ"]
channel_lambdas = 299792458 / channel_freqs

# now we want to get the uv positions (in lambdas) for each visibility sample
uvws = t.getcol('UVW') # in units of metres
uvws_l = np.repeat(uvws[:, :, np.newaxis], len(channel_lambdas), axis=2)    # duplicate each uv sample into a size such that we can multiply with the channel lambdas
uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
chan_tiled = np.ones_like(uvs_l)*channel_lambdas[None, None, :] # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
uvs_l = uvs_l / chan_tiled    # get the uvw positions in terms of the number of lambda instead of metres

# get the time for each uv sample
times = np.repeat(vis_time, len(channel_freqs))

N_pixels = 320  # number of pixels for the uv grid and will eventually be (N_pixels/2)xN_pixels for the dynamic grid precursor
max_uv = np.max(np.abs(uvs_l)) # our grid needs to extend to the positive and negative max_uv value on both axes
uv_grid = np.zeros((N_pixels, N_pixels))
bins = np.linspace(-max_uv, max_uv, N_pixels)
uvcell_size = bins[1] - bins[0]

print("max uv distance =", max_uv)

# now get the uv grid indices for each uv sample
uv_indices = np.round(uvs_l/uvcell_size).astype(int) + N_pixels//2
u_indices = uv_indices[:, 0, :]
v_indices = uv_indices[:, 1, :]

flags = t.getcol('FLAG')
visibility_data = np.ma.array(t.getcol('DATA'), mask=(flags == True))
# for now we'll work with XX and YY data separately
visibilities_xx = visibility_data[:, :, 0]
visibilities_yy = visibility_data[:, :, -1]

uv_f_t_xx = np.ma.array(np.zeros((N_pixels, N_pixels, len(channel_lambdas), len(unique_times)), dtype=np.complex64), mask=False)    # set up complex uv grid for our visibilities
weights_f_t_xx = np.zeros((N_pixels, N_pixels, len(channel_lambdas), len(unique_times)))
uv_f_t_yy = np.ma.array(np.zeros((N_pixels, N_pixels, len(channel_lambdas), len(unique_times)), dtype=np.complex64), mask=False)   # set up complex uv grid for our visibilities
weights_f_t_yy = np.zeros((N_pixels, N_pixels, len(channel_lambdas), len(unique_times)))

print("Forming uv-grid cells through time...")
allchan_inds=np.arange(len(channel_lambdas), dtype=int)

for it in tqdm(range(len(unique_times))):
    # looping over each unique time, and populating the uv grid for that time with visibilities:
    # we need to find the indices that correspond to the current time
    # if it == len(unique_times) - 1:
    #     valid_times = unique_times[it] <= vis_time
    # else:
    #     valid_times = (( vis_time >= unique_times[it] ) & ( vis_time < unique_times[it + 1]))

    valid_times = vis_time == unique_times[it]

    # tsample = unique_times[it]
    # t1 = taql("select * from $t where TIME_CENTROID == $tsample")
    # print(t1.getcol('FLAG').shape)
    # vis_flags_xx = t1.getcol('FLAG')[:, :, 0]
    # vis_flags_yy = t1.getcol('FLAG')[:, :, 3]

    # unflagged_xx_ = vis_flags_xx == False
    # unflagged_yy_ = vis_flags_yy == False

    # vis_xx_ =  t1.getcol('DATA')[:, :, 0]
    # vis_yy_ =  t1.getcol('DATA')[:, :, 3]

    # #forcibly set flagged data to 0
    # vis_xx_[~unflagged_xx_] = 0.0+1j*0.0
    # vis_yy_[~unflagged_yy_] = 0.0+1j*0.0

    vis_xx_ = visibilities_xx[valid_times]
    vis_yy_ = visibilities_yy[valid_times]

    u_inds_xx_ =  u_indices[valid_times]#[unflagged_xx_]
    u_inds_yy_ =  u_indices[valid_times]#[unflagged_yy_]
    v_inds_xx_ =  v_indices[valid_times]#[unflagged_xx_]
    v_inds_yy_ =  v_indices[valid_times]#[unflagged_yy_]
    #chan_inds_xx_ = allchan_inds_#[unflagged_xx_]
    #chan_inds_yy_ = allchan_inds_#[unflagged_yy_]

    # print(u_indices.shape, u_inds_xx_.shape)
    # print(uv_f_t_xx.shape, vis_xx_.shape)
    # break

    ## and we use the following numpy operation to add samples to the correct indices (even accounting for duplicate indices!)
    # unflagged_xx_.astype(int) casts flag True/False values to 1s and 0s
    np.add.at(weights_f_t_xx, (u_inds_xx_, v_inds_xx_, allchan_inds, it), 1)    # use this to see the weighting of each uv cell
    np.add.at(weights_f_t_yy, (u_inds_yy_, v_inds_yy_, allchan_inds, it), 1)    # use this to see the weighting of each uv cell

    np.add.at(uv_f_t_xx, (u_inds_xx_, v_inds_xx_, allchan_inds, it), vis_xx_)
    np.add.at(uv_f_t_yy, (u_inds_yy_, v_inds_yy_, allchan_inds, it), vis_yy_)

    # t1.close()

uv_f_t = uv_f_t_xx + uv_f_t_yy
print("summed xx and yy")

# sum up the gridded visibility cube along the 3rd axis
# corresponding to freq
# note this is equivalent to DM=0.
uv_t = np.sum(uv_f_t, axis=2)
print("summed over freq")

# now split the array into two long blocks so that we can flip the bottom in uv space and add it as a complex conjugate to the top block
top, bottom = np.vsplit(uv_t, 2)
print('split array')

uv_t_half = top + np.conjugate(bottom[::-1, ::-1, :])
print('collapsed uv space')

uv_t_half_allT = np.sum(np.abs(uv_t_half), axis=2) # sum the absolute values in case some numbers would cancel each other out otherwise
print('summed over time')

nz_rows, nz_cols = np.nonzero(uv_t_half_allT)

dynamic_grid = np.zeros((len(nz_rows), len(unique_times)), dtype=np.complex64)

print("Populating dynamic grid...")
for it in tqdm(range(len(unique_times))):
    dynamic_grid[:, it] = uv_t[nz_rows, nz_cols, it]

np.save('dynamic_grid.npy', dynamic_grid)

unique_indices = np.array([nz_rows, nz_cols])
np.save('unique_indices.npy', unique_indices)

weights_t = np.sum(weights_f_t_xx + weights_f_t_yy, axis=2)
np.save('weights_t.npy', weights_t)

print(time.time() - time1)
