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
inds = np.argwhere(np.abs(np.sum(uvws, axis=1)) > 150.0).squeeze()
t = t.selectrows(inds)

# now get rid of data on the longest baselines (antenna2 >= 30 are removed)
# we get rid of those large baselines as they are sparsely sampled in uv space and so just adds unnecessary computational cost for baselines above ~9k lambda
antenna2 = t.getcol('ANTENNA2')
rownrs = np.arange(len(antenna2))
use_baselines = np.where(antenna2 < 30)
t = t.selectrows(rownrs[use_baselines])

tf = table(f"{ms}/SPECTRAL_WINDOW")
# ta = table(f"{ms}/ANTENNA")
vis_time = t.getcol('TIME')
vis_feed = t.getcol('FEED1')
vis_scan = t.getcol('SCAN_NUMBER')

# get each unique timestamp 
unique_times = np.unique(vis_time)
unique_times -= unique_times[0]

# get all channel frequencies and convert that to wavelengths
channel_freqs = tf[0]["CHAN_FREQ"]
channel_lambdas = 299792458 / channel_freqs

# now we want to get the uv positions (in lambdas) for each visibility sample
uvws = t.getcol('UVW') # in units of metres
uvws_l = np.repeat(uvws[:, :, np.newaxis], len(channel_lambdas), axis=2)    # duplicate each uv sample into a size such that we can multiply with the channel lambdas
uvs_l = uvws_l[:, :2, :]        # gets rid of w axis. DONT FORGET: need to do w-projection! can't just ignore the w axis like im doing here
chan_tiled = np.repeat(np.repeat(channel_lambdas[np.newaxis, :], uvs_l.shape[1], axis=0)[np.newaxis, :, :], uvs_l.shape[0], axis=0) # arrange and repeat the channel lambdas in an array shape such that we can multiply it onto the uv samples
uvs_l = uvs_l / chan_tiled    # get the uvw positions in terms of the number of lambda instead of metres

# get the time for each uv sample
times = np.repeat(vis_time, len(channel_freqs))
times -= times[0]
# flatten uv coords across baselines and time
uvs_l = np.array([uvs_l[:, 0, :].flatten(), uvs_l[:, 1, :].flatten()])
uvs_l = np.swapaxes(uvs_l, 0, 1)

N_pixels = 320  # number of pixels for the uv grid and will eventually be (N_pixels/2)xN_pixels for the dynamic grid precursor
max_uv = np.max(np.abs(uvs_l)) # our grid needs to extend to the positive and negative max_uv value on both axes
uv_grid = np.zeros((N_pixels, N_pixels))
bins = np.linspace(-max_uv, max_uv, N_pixels)

print("max uv distance =", max_uv)

# now get the uv grid indices for each uv sample
u_indices = np.digitize(uvs_l[:, 0], bins)
v_indices = np.digitize(uvs_l[:, 1], bins)

# visibility_data = t.getcol('DATA')
flags = t.getcol('FLAG')
visibility_data = np.ma.array(t.getcol('DATA'), mask=(flags == True))
visibilities_xx_yy = visibility_data[:, :, 0] + visibility_data[:, :, -1]    # for now we'll work with the XX+YY data
visibilities_xx_yy = visibilities_xx_yy.flatten(order='C')                  # and finally we need to flatten the visibilities across channels into the correct format (t1: C1, C2..., t2: C1, C2..., ...)
uv_t = np.zeros((N_pixels, N_pixels, len(unique_times)), dtype=np.complex64)    # set up complex uv grid for our visibilities

print("Forming uv-grid cells through time...")

for it in tqdm(range(len(unique_times))):
    # looping over each unique time, and populating the uv grid for that time with visibilities:
    # we need to find the indices that correspond to the current time
    if it == len(unique_times) - 1:
        valid_times = unique_times[it] <= times
    else:
        valid_times = ((unique_times[it] <= times) & (times < unique_times[it + 1]))

    ## and we use the following numpy operation to add samples to the correct indices (even accounting for duplicate indices!)
    # np.add.at(uv_t, (u_indices[valid_times], v_indices[valid_times], it), 1)    # use this to see the weighting of each uv cell
    np.add.at(uv_t, (u_indices[valid_times], v_indices[valid_times], it), visibilities_xx_yy[valid_times]) # use this to add the actual visibilities

# now split the array into two long blocks so that we can flip the bottom in uv space and add it as a complex conjugate to the top block
top, bottom = np.vsplit(uv_t, 2)

uv_t_half = top + np.conjugate(bottom[::-1, ::-1, :])

uv_t_half_allT = np.sum(np.abs(uv_t_half), axis=2) # sum the absolute values in case some numbers would cancel each other out otherwise

nz_rows, nz_cols = np.nonzero(uv_t_half_allT)

dynamic_grid = np.zeros((len(nz_rows), len(unique_times)), dtype=np.complex64)

print("Populating dynamic grid...")
for it in tqdm(range(len(unique_times))):
    dynamic_grid[:, it] = uv_t_half[nz_rows, nz_cols, it]


np.save('dynamic_grid.npy', dynamic_grid)

unique_indices = np.array([nz_rows, nz_cols])
print(unique_indices.shape)
np.save('unique_indices.npy', unique_indices)

print(time.time() - time1)


# uv_t_flat_ = uv_t.flatten()
# uv_t_flat = uv_t_flat_[uv_t_flat_>0]

# plt.hist(uv_t.flatten(),bins=300)
# plt.yscale('log')
# plt.savefig('1dhist.png', dpi=400)
# plt.close() 

# with np.printoptions(threshold=np.inf):
#     print(uv_t[:100, 2000:, 0])

np.save('uvt.npy', uv_t)




### MOVIE of dynamic grid scrolling
# fig, ax = plt.subplots(figsize=(4, 40))

# temp_array = 0.5 * (dynamic_grid[0::4] + dynamic_grid[1::4] + dynamic_grid[2::4] + dynamic_grid[3::4])
# ax.imshow(np.abs(temp_array), norm='log', interpolation='none')
# fig.savefig('dynamic_grid.png', dpi=400)

# from matplotlib import animation
# delta_y = len(nz_rows) // 250
# every = 12
# length = 25
# frames = np.arange(0, len(temp_array[:, 0]) - delta_y, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
# fps = len(frames) // length  # fps for the final animation

# fig, ax = plt.subplots(figsize=(6, 6 * 250 / len(unique_times)))
# ax.set_axis_off()
# fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
# pbar = tqdm(total=len(frames))
# pbar.update(1)

# ax.imshow(np.abs(temp_array), norm='log', interpolation='none')

# def animate(i):
#     # update progressbar for movie
#     if i > 0:   # for some reason matplotlib runs the 0th iteration like 3 times or something. this means we dont update the progress bar each 0th iter
#         pbar.update(1)
#     ax.set(ylim=(i + delta_y, i), xlabel='Time Sample', ylabel='uv Cell No.')
#     return fig, 

# ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
# ani.save(f"dynamic_grid_movie.gif", writer='ffmpeg', fps=fps) 
# pbar.close()





fig, ax = plt.subplots()


# #uv_t[:,:,0][uv_t[:,:,0] == 0]=np.nan
ax.imshow(np.abs(uv_t_half[:, :, 0]), norm='log', interpolation='none')
# ax.plot(u_indices[valid_times], v_indices[valid_times], markersize=0.1, c='k', marker='.', lw=0)
fig.savefig('plot2_test_half.png', dpi=400)

fig, ax = plt.subplots()
ax.imshow(np.abs(uv_t[:, :, 0]), norm='log', interpolation='none')
# ax.plot(u_indices[valid_times], v_indices[valid_times], markersize=0.1, c='k', marker='.', lw=0)
fig.savefig('plot2_test.png', dpi=400)








# # # print(uvws_l.shape, chan_tiled.shape, h.shape)


# # # fig, ax = plt.subplots(figsize=(8, 8))

# # # ax.plot(uvs_l[:, 0], uvs_l[:, 1], ls='', ms=0.1, marker='.', alpha=0.5)
# # # ax.set(xlabel=r"v ($\lambda$)", ylabel=r'u ($\lambda$)')

# # # fig.savefig('plot.png', dpi=400)
# # # 

# # # print(data.shape, uvs_l.shape, len(times), len(vis_time))


# # # start with XX data only for now










