from matplotlib import animation
import numpy as np
import riptide as rt
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
#h
#DGA = np.load('dga.npy')
dga_fn = 'newdata/scienceData.TEST_0900-40.SB41912.TEST_0900-40.beam15_averaged_cal.leakage.uvsub4.ps2.dga.npy'
N_pixels = 384
#newdata/scienceData.TEST_0900-40.SB41912.TEST_0900-40.beam15_averaged_cal.leakage.dga.npy'
#'newdata/scienceData.TEST_0900-40.SB41912.TEST_0900-40.beam15_averaged_cal.leakage.uvsub4.dga.npy'
#scienceData.TEST_0900-40.SB41912.TEST_0900-40.beam15_averaged_cal.leakage.uvsub4.ps.dga.npy
#newdata/scienceData.TEST_0900-40.SB41912.TEST_0900-40.beam15_averaged_cal.leakage.uvsub3.ps.dga.npy'
DGA = np.load(dga_fn)#dga3_fullfield.npy')
#print(DGA.shape)
#DGA = np.load('dga3_fullfield.npy')#, dga_all)
unique_indices = DGA[1:, 0]
unique_times = DGA[0, 1:]

print(unique_indices)
DGA = DGA[1:, 1:]

fig, ax = plt.subplots()
ax.plot(np.sum(DGA.real, axis=0), label='real')
ax.plot(np.sum(DGA.imag, axis=0), label='imag')
ax.legend()
fig.savefig(f'{dga_fn.replace(".npy", "")}.lightcurve.png', dpi=400)
plt.show()

DGA__ = DGA.copy()
#DGA__[np.abs(DGA__) < 0.0001] = np.nan*1j
mean_DGA = np.sum(DGA__, axis=1) #take sum along time axis doofus
nonzero_DGA = np.sum((np.abs(DGA__) > 1e-4), axis=1) #number of nonzero entries in each row
#plt.plot(nonzero_DGA)
#plt.show()
nonzero_DGA[nonzero_DGA == 0] = 1 #fix up div by 0 error
mean_DGA /= nonzero_DGA
#DGA = DGA -  mean_DGA[:, None]
#DGA__ = None

#unique_indices = np.load('uv_indices3.npy')


#DGA_ = DGA.copy()
#DGA_[(np.abs(DGA_) < 1e-5)+(np.abs(DGA_) > 100.0)] = np.nan
#mean_DGA = np.nanmedian(DGA_,axis=1)
#DGA -= mean_DGA[:, None]
#unique_indices = np.load('uv_indices.npy')
#unique_indices = np.load('uv_indices2.npy')#[:2100]
#unique_indices = np.load('uv_indices3_fullfield.npy')#, alluv_inds)
noiseamp = np.random.normal(loc=0.0, scale=250.0, size=DGA.shape)
noisephase = np.random.uniform(low=-np.pi, high=np.pi, size=DGA.shape)
noise=noiseamp * np.exp(1j*noisephase)
#DGA += noise
print(len(unique_indices))


def sigma_clip(arr):
    sigma = 99999
    arr = arr.copy()
    arr[np.abs(arr) > 3.0 * sigma] = np.nan
    for i in range(15):
        sigma = np.nanstd(arr)
        arr[arr > 3.0*sigma] = np.nan
        
    return sigma

t_samples = DGA.shape[1]

mean_uv_grid = np.zeros((N_pixels,N_pixels), dtype=complex)

np.add.at(mean_uv_grid, (unique_indices.real.astype(int), unique_indices.imag.astype(int)), mean_DGA)
np.add.at(mean_uv_grid, (N_pixels - unique_indices.real.astype(int), N_pixels - unique_indices.imag.astype(int)), np.conj( mean_DGA))
mean_uv_grid[np.isnan(mean_uv_grid)] = 0.0 + 0.0*1j

mean_uv_snapshot = np.fft.fftshift(mean_uv_grid)# + np.conjugate(uv_grid[::-1, ::-1, i])
mean_fft = np.fft.fftshift(np.fft.ifft2(mean_uv_snapshot))
plt.imshow(np.real(mean_fft[:, ::-1]).T, interpolation='none', origin='lower', aspect='equal', cmap='inferno', extent=[-0.55, 0.55, -0.55, 0.55])

plt.show()            

print(t_samples)
period=2656.247
#period = 110.2
period = 75.88554711
tsamp = 10.0
phase = ((unique_times - unique_times[0]).real % period) / period
base_period = int(period/tsamp)
phaseint = np.round(phase * base_period).astype(int)
print(np.unique(phaseint))
folded = []
for i, p in enumerate(np.unique(phaseint)):
    timesamps = np.argwhere(phaseint == p).squeeze()
    dga_p = DGA[None, timesamps]
    dga_p = dga_p[np.abs(dga_p) > 1e-5]
    folded.append(np.mean(dga_p))

folded = np.array(folded)
plt.plot(np.unique(phaseint), folded)
plt.show()
#sys.exit()
#period=120
#period=2923.4
tsamp=np.median(np.diff(unique_times))#sec
print(tsamp)
base_period = int(period/tsamp)

periods = rt.libffa.ffaprd(t_samples, base_period, dt=tsamp)
#print(periods)
#print(periods[6])

print(len(periods), periods)

cube = np.zeros((DGA.shape[0], len(periods), base_period), dtype=complex)

for i in range(DGA.shape[0]):
    ffa_real = rt.libffa.ffa1(np.ascontiguousarray(DGA[i, :].real), base_period)
    ffa_imag = rt.libffa.ffa1(np.ascontiguousarray(DGA[i, :].imag), base_period)
    if i == 0:
        print(ffa_real.shape)
    #ffa = rt.libffa.ffa1(np.ascontiguousarray(DGA[i, :], base_period))
    #print(ffa[0,0])
    cube[i, :, :] = ffa_real + 1j * ffa_imag

uv_grid = np.zeros((N_pixels, N_pixels, base_period ), dtype=complex)

j=int(np.argmin(np.abs(periods-period)).squeeze())#0.6*len(periods))
print(j)
print(periods[j])


plt.plot(np.sum(cube[:, j, :],axis=0).real)
plt.show()
print(j, periods[j])
for i in range(base_period):
#   for j in range(len(periods)):
    np.add.at(uv_grid, (unique_indices.real.astype(int), unique_indices.imag.astype(int), i), cube[:, j, i])
    np.add.at(uv_grid, (N_pixels - unique_indices.real.astype(int), N_pixels - unique_indices.imag.astype(int), i), np.conj(cube[:, j, i]))


uv_snapshots = np.zeros((N_pixels, N_pixels, base_period), dtype=complex)
uv_angle = np.zeros((N_pixels, N_pixels, base_period))
ffts = np.zeros(uv_snapshots.shape, dtype=complex)
print(ffts.shape)
for i in range(base_period):
    #uv_snapshots[:, :, i] = np.vstack((uv_grid[:, :, i], np.conjugate(uv_grid[::-1, ::-1, i])))
    uv_snapshots[:, :, i] = np.fft.fftshift(uv_grid[:, :, i]) #+ np.conjugate(uv_grid[::-1, ::-1, i])
    uv_angle[:, :, i] = np.fft.fftshift(np.angle(uv_snapshots[:, :, i], deg=True))
    ffts[:, :, i] = np.fft.fftshift(np.fft.ifft2(uv_snapshots[:, :, i]))     # need to fftshift to avoid putting the phase center at the corners

uv_angle[uv_angle == 0.0] = np.nan
    


def animate(i):
    # update progressbar for movie
    if i > 0:   # for some reason matplotlib runs the 0th iteration like 3 times or something. this means we dont update the progress bar each 0th iter
        pbar.update(1)
    ax[0].cla(); ax[1].cla()
    
    # ax[0].imshow(np.abs(uv_snapshots[:, :, i]), norm='log', interpolation='none', vmin=np.min(abs(uv_snapshots)), vmax=np.max(abs(uv_snapshots)))
    ax[0].imshow(uv_angle[:, :, i], interpolation='none', origin='lower',clim=(-1,1))#np.min(abs(uv_angle)), vmax=np.max(abs(uv_angle)))
    ax[1].imshow(np.real(ffts[:, ::-1, i]).T, origin='lower', interpolation='none', aspect='equal')#clim=(10,100))#np.min(abs(ffts)), vmax=np.max(abs(ffts)))

    ax[0].set(title=f'Frame {i}/{base_period}')

    return fig, 
# fig, axes = plt.subplots(ncols=2, nrows=2)
# def animate(i):
#     # update progressbar for movie
#     if i > 0:   # for some reason matplotlib runs the 0th iteration like 3 times or something. this means we dont update the progress bar each 0th iter
#         pbar.update(1)
#     for ax in axes.flatten():
#         ax.cla()
    
#     axes[0, 0].imshow(uv_snapshots[:, :, i].real, interpolation='none', vmin=np.min(uv_snapshots.real), vmax=np.max(uv_snapshots.real))
#     axes[1, 0].imshow(uv_snapshots[:, :, i].imag, interpolation='none', vmin=np.min(uv_snapshots.imag), vmax=np.max(uv_snapshots.imag))
#     axes[0, 1].imshow(ffts[:, :, i].real, vmin=np.min(ffts.real), vmax=np.max(ffts.real))
#     axes[1, 1].imshow(ffts[:, :, i].imag, vmin=np.min(ffts.imag), vmax=np.max(ffts.imag))

#     return fig, 



every = 1
length = 0.8
print(base_period)

frames = np.arange(0, base_period, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
plt.close()
for f in frames:
    
    plt.imshow(np.real(ffts[:, ::-1, f]).T, origin='lower', interpolation='none', aspect='equal')
    plt.title(f)
    plt.show()
    
print(frames)
fps = len(frames) / length  # fs for the final animation
#fps = 3
pbar = tqdm(total=len(frames))
pbar.update(1)
fig, ax = plt.subplots(ncols=2)
ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)#, bbox_inches='tight')
ani.save(f"temp_0900.gif", writer='ffmpeg', fps=fps) 
ani.save(f"temp_0900.mp4", writer='ffmpeg', fps=fps) 
pbar.close()

#sys.exit()    
#pbar = tqdm(total=len(frames))
#pbar.update(1)
mean_uv_grid = np.zeros((N_pixels,N_pixels), dtype=complex)

np.add.at(mean_uv_grid, (unique_indices.real.astype(int), unique_indices.imag.astype(int)), mean_DGA)
np.add.at(mean_uv_grid, (N_pixels - unique_indices.real.astype(int), N_pixels - unique_indices.imag.astype(int)), np.conj( mean_DGA))
mean_uv_grid[np.isnan(mean_uv_grid)] = 0.0 + 0.0*1j

uv_snapshots = np.zeros((N_pixels, N_pixels, base_period), dtype=complex)
uv_angle = np.zeros((N_pixels, N_pixels, base_period))
ffts = np.zeros(uv_snapshots.shape, dtype=complex)
uv_amp = np.zeros((N_pixels, N_pixels, base_period))
for i in (range(base_period)):
    for j in tqdm(range(len(periods))):
        #if i % 20 == 0:
        #    print(i,j)
        uv_grid = np.zeros((N_pixels, N_pixels ), dtype=complex)

        np.add.at(uv_grid, (unique_indices.real.astype(int), unique_indices.imag.astype(int)), cube[:, j, i])
        np.add.at(uv_grid, (N_pixels - unique_indices.real.astype(int), N_pixels - unique_indices.imag.astype(int)), np.conj(cube[:, j, i]))
        #print(uv_grid.shape)
        #print(mean_uv_grid.shape)

        #uv_grid[np.abs(uv_grid) > 1e-5] -= mean_uv_grid[np.abs(uv_grid) > 1e-5]
        
        uv_snapshots = np.fft.fftshift(uv_grid)# + np.conjugate(uv_grid[::-1, ::-1, i])
        #uv_angle = np.angle(uv_snapshots, deg=True)

        #uv_amp = np.abs(uv_snapshots)
    
        ffts = np.fft.fftshift(np.fft.ifft2(uv_snapshots))
        sigma_ = sigma_clip(ffts)
        #print(sigma_)
        # need to fftshift to avoid putting the phase center at the corners
        #plt.imshow(np.real(ffts), interpolation='none', origin='lower', aspect='equal', cmap='inferno')#, extent=[-0.55, 0.55, -0.55, 0.55])
        #plt.show()
        if np.max(ffts.real) > 5.0 * sigma_ :
            print("CAND FOUND")
            index = np.unravel_index(np.argmax(ffts.real), shape = ffts.real.shape)
            print(i, j, periods[j], base_period, np.max(ffts.real), index)
            plt.imshow(np.real(ffts), interpolation='none', origin='lower', aspect='equal', cmap='inferno')#, extent=[-0.55, 0.55, -0.55, 0.55])
            plt.show()


mean_uv_snapshot = np.fft.fftshift(mean_uv_grid)# + np.conjugate(uv_grid[::-1, ::-1, i])
mean_fft = np.fft.fftshift(np.fft.ifft2(mean_uv_snapshot))
plt.imshow(np.real(mean_fft), interpolation='none', origin='lower', aspect='equal', cmap='inferno', extent=[-0.55, 0.55, -0.55, 0.55])

plt.show()            

uv_amp[np.abs(uv_snapshots) == 0.0] = np.nan
uv_angle[np.abs(uv_snapshots) == 0.0] = np.nan

fig, ax = plt.subplots()
# array = np.abs(np.sum(cube, axis=0))
array = np.sum(np.abs(cube), axis=0)
ax.imshow(array,origin='lower',aspect='auto',interpolation='none')#,clim=(0, 30))
fig.savefig(f'{dga_fn.replace(".npy", "")}.butterfly.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()


print(np.unravel_index(np.argmax(array), array.shape), np.max(array))
print(np.unravel_index(np.argmin(array), array.shape), np.min(array))
topinds_ = np.argpartition(array.flatten(), -10)
topinds = topinds_[np.argsort(array.flatten()[topinds_])]
topinds = np.unravel_index(topinds, array.shape)
for i in range(100):
    print(topinds[0][i], topinds[1][i])

plt.plot(array[topinds[0][-1]])
plt.show()


# fig, axs = plt.subplots(ncols=3, nrows=2, sharex= True, sharey=True)
# axs[0,0].imshow((ffts[:, :, 35]).real, origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[0,0].set_title('Real')
# axs[0,1].imshow((ffts[:, :, 35]).imag, origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[0,1].set_title('Imag')
# axs[1,0].imshow((ffts[:, :, 23]).real,origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[1,1].imshow((ffts[:, :, 23]).imag,origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')


# axs[0,2].imshow(np.abs(ffts[:, :, 35]).real, origin='lower',aspect='auto',interpolation='none', clim=(0,5), cmap='inferno')
# axs[1,2].imshow(np.abs(ffts[:, :, 23]).real, origin='lower',aspect='auto',interpolation='none', clim=(0,5),cmap='inferno')
# axs[0,2].set_title('Abs')

# print('that was snr')
# fig.savefig('frame35.png', dpi=400)
# plt.show()

#sigma19 = sigma_clip(ffts[:, :, 217].real)
#peak19 = np.max((ffts[:, :, 217]).real)
#print(peak19/sigma19)
# fig, axs = plt.subplots(ncols=3, nrows=2, sharex= True, sharey=True)
# axs[0,0].imshow((ffts[:, :, 19]).real, origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[0,0].set_title('Real')
# axs[0,1].imshow((ffts[:, :, 19]).imag, origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[0,1].set_title('Imag')
# axs[1,0].imshow((ffts[:, :, 13]).real,origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[1,1].imshow((ffts[:, :, 13]).imag,origin='lower',aspect='auto',interpolation='none', clim=(-7,7), cmap='bwr')
# sigma19 = sigma_clip(ffts[:, :, 19].real)
# peak19 = np.max((ffts[:, :, 19]).real)

# axs[0,2].imshow(np.abs(ffts[:, :, 19]).real, origin='lower',aspect='auto',interpolation='none', clim=(0,5), cmap='inferno')
# axs[1,2].imshow(np.abs(ffts[:, :, 13]).real, origin='lower',aspect='auto',interpolation='none', clim=(0,5),cmap='inferno')
# axs[0,2].set_title('Abs')

# print(peak19/sigma19)
# print('that was snr')
# fig.savefig('frame19.png', dpi=400)
# plt.show()

# fig, axs = plt.subplots(ncols=3, nrows=2, sharex= True, sharey=True)
# axs[0,0].imshow((ffts[:, :, 201]).real, origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[0,0].set_title('Real')
# axs[0,1].imshow((ffts[:, :, 201]).imag, origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[0,1].set_title('Imag')
# axs[1,0].imshow((ffts[:, :, 162]).real,origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
# axs[1,1].imshow((ffts[:, :, 162]).imag,origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
# sigma19 = sigma_clip(ffts[:, :, 217].real)
# peak19 = np.max((ffts[:, :, 217]).real)

# axs[0,2].imshow(np.abs(ffts[:, :, 201]).real, origin='lower',aspect='equal',interpolation='none', clim=(0,5), cmap='inferno')
# axs[1,2].imshow(np.abs(ffts[:, :, 162]).real, origin='lower',aspect='equal',interpolation='none', clim=(0,5),cmap='inferno')
# axs[0,2].set_title('Abs')

# print(peak19/sigma19)
# print('that was snr')
# fig.savefig('frame201.png', dpi=400)
# plt.show()


# fig,ax = plt.subplots(1,1,figsize=(8,8))
# ax.imshow((ffts[:, :, 201]).real, origin='lower',aspect='equal',interpolation='none', cmap='viridis', clim=(-1,7))
# plt.axis('off')
# plt.savefig('frame201_real.png', dpi=400, bbox_inches='tight')


#fig, ax = plt.subplots()

sigma16 = sigma_clip(ffts[:, :, 162].real)
peak16 = np.max((ffts[:, :, 162]).real)

print(peak16/sigma16)
print('that was snr for otherpeak')
                    
#fig.savefig('frame15.png', dpi=400)
#plt.show()
plt.close()


