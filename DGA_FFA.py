from matplotlib import animation
import numpy as np
import riptide as rt
import matplotlib.pyplot as plt
#from tqdm import tqdm


#DGA = np.load('dga.npy')
DGA = np.load('dga3.npy')
#DGA = np.load('dga3_fullfield.npy')#, dga_all)
unique_indices = DGA[1:, 0]
unique_times = DGA[0, 1:]
DGA = DGA[1:, 1:]


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
#fig, ax = plt.subplots()
#ax.plot(np.sum(DGA.real, axis=0), label='real')
#ax.plot(np.sum(DGA.imag, axis=0), label='imag')
#ax.legend()
#fig.savefig('lightcurve.png', dpi=400)


def sigma_clip(arr):
    sigma = 99999
    arr = arr.copy()
    arr[np.abs(arr) > 3.0 * sigma] = np.nan
    for i in range(15):
        sigma = np.nanstd(arr)
        arr[arr > 3.0*sigma] = np.nan
        
    return sigma

t_samples = DGA.shape[1]

print(t_samples)
period=2656.247
#period=2923.4
tsamp=10.0 #sec
base_period = int(period/tsamp)

periods = rt.libffa.ffaprd(t_samples, base_period, dt=tsamp)
print(periods)
print(periods[6])

print(len(periods), periods)

cube = np.zeros((DGA.shape[0], len(periods), base_period), dtype=complex)

for i in range(DGA.shape[0]):
    ffa_real = rt.libffa.ffa1(np.ascontiguousarray(DGA[i, :].real), base_period)
    ffa_imag = rt.libffa.ffa1(np.ascontiguousarray(DGA[i, :].imag), base_period)
    #ffa = rt.libffa.ffa1(np.ascontiguousarray(DGA[i, :], base_period))
    #print(ffa[0,0])
    cube[i, :, :] = ffa_real + 1j * ffa_imag

uv_grid = np.zeros((320, 320, base_period, len(periods) ), dtype=complex)

#for i in range(base_period):
#    for j in range(len(periods)):
#        np.add.at(uv_grid, (unique_indices.real.astype(int), unique_indices.imag.astype(int), i, j), cube[:, j, i])
#        np.add.at(uv_grid, (320 - unique_indices.real.astype(int), 320 - unique_indices.imag.astype(int), i), np.conj(cube[:, j, i]))

#pbar = tqdm(total=len(frames))
#pbar.update(1)

uv_snapshots = np.zeros((320, 320, base_period), dtype=complex)
uv_angle = np.zeros((320, 320, base_period))
ffts = np.zeros(uv_snapshots.shape, dtype=complex)
uv_amp = np.zeros((320, 320, base_period))
for i in range(base_period):
    for j in range(len(periods)):
        uv_grid = np.zeros((320, 320 )), dtype=complex)

        np.add.at(uv_grid, (unique_indices.real.astype(int), unique_indices.imag.astype(int), i, j), cube[:, j, i])
        np.add.at(uv_grid, (320 - unique_indices.real.astype(int), 320 - unique_indices.imag.astype(int), i), np.conj(cube[:, j, i]))

        
        uv_snapshots[:, :, i] = np.fft.fftshift(uv_grid[:, :, i])# + np.conjugate(uv_grid[::-1, ::-1, i])
        uv_angle[:, :, i] = np.angle(uv_snapshots[:, :, i], deg=True)

        uv_amp[:, :, i] = np.abs(uv_snapshots[:, :, i])
    
        ffts[:, :, i] = np.fft.fftshift(np.fft.ifft2(uv_snapshots[:, :, i]))     # need to fftshift to avoid putting the phase center at the corners
        if np.max(ffts[:, :, i].real) > 10.0 * sigma_clip(ffts[:, :, i]):
            print(i, base_period)
            plt.imshow(np.real(ffts[:, :, i]), interpolation='none', origin='lower', aspect='equal', cmap='inferno', extent=[-0.55, 0.55, -0.55, 0.55])
        plt.show()
        

uv_amp[np.abs(uv_snapshots) == 0.0] = np.nan
uv_angle[np.abs(uv_snapshots) == 0.0] = np.nan

fig, ax = plt.subplots()
# array = np.abs(np.sum(cube, axis=0))
array = np.sum(np.abs(cube), axis=0)
ax.imshow(array,origin='lower',aspect='auto',interpolation='none')#,clim=(0, 30))
fig.savefig('butterfly.png', dpi=400, bbox_inches='tight')
plt.show()
plt.close()


print(np.unravel_index(np.argmax(array), array.shape), np.max(array))
print(np.unravel_index(np.argmin(array), array.shape), np.min(array))
topinds_ = np.argpartition(array.flatten(), -100)
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

fig, axs = plt.subplots(ncols=3, nrows=2, sharex= True, sharey=True)
axs[0,0].imshow((ffts[:, :, 201]).real, origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
axs[0,0].set_title('Real')
axs[0,1].imshow((ffts[:, :, 201]).imag, origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
axs[0,1].set_title('Imag')
axs[1,0].imshow((ffts[:, :, 162]).real,origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
axs[1,1].imshow((ffts[:, :, 162]).imag,origin='lower',aspect='equal',interpolation='none', clim=(-7,7), cmap='bwr')
sigma19 = sigma_clip(ffts[:, :, 217].real)
peak19 = np.max((ffts[:, :, 217]).real)

axs[0,2].imshow(np.abs(ffts[:, :, 201]).real, origin='lower',aspect='equal',interpolation='none', clim=(0,5), cmap='inferno')
axs[1,2].imshow(np.abs(ffts[:, :, 162]).real, origin='lower',aspect='equal',interpolation='none', clim=(0,5),cmap='inferno')
axs[0,2].set_title('Abs')

print(peak19/sigma19)
print('that was snr')
fig.savefig('frame201.png', dpi=400)
plt.show()


fig,ax = plt.subplots(1,1,figsize=(8,8))
ax.imshow((ffts[:, :, 201]).real, origin='lower',aspect='equal',interpolation='none', cmap='viridis', clim=(-1,7))
plt.axis('off')
plt.savefig('frame201_real.png', dpi=400, bbox_inches='tight')


#fig, ax = plt.subplots()

sigma16 = sigma_clip(ffts[:, :, 162].real)
peak16 = np.max((ffts[:, :, 162]).real)

print(peak16/sigma16)
print('that was snr for otherpeak')
                    
#fig.savefig('frame15.png', dpi=400)
#plt.show()
plt.close()


fig, ax = plt.subplots(figsize=(13,4),ncols=3, layout="constrained")

def animate(i):
    # update progressbar for movie
    ax[0].cla(); ax[1].cla()

    ax[0].imshow(np.fft.fftshift(uv_angle[:, :, i]), clim=(-180, 180), interpolation='none', origin='lower', aspect='equal', cmap='inferno', extent=[-8700, 8700, -8700, 8700])
    ax[1].imshow(np.fft.fftshift(uv_amp[:, :, i]), clim=(0, 20), interpolation='none', origin='lower', aspect='equal', cmap='inferno', extent=[-8700, 8700, -8700, 8700])
    ax[2].imshow(np.real(ffts[:, :, i]), clim=(-1,7), interpolation='none', origin='lower', aspect='equal', cmap='inferno', extent=[-0.55, 0.55, -0.55, 0.55])
    

    ax[0].set(title=f'Phase', xlabel=r'$u/\lambda$', ylabel=r'$v/\lambda$')
    ax[1].set(title='Amp', xlabel=r'$u/\lambda$', ylabel=r'$v/\lambda$')
    ax[2].set(title='Image', xlabel=r'$\Delta$RA (deg)', ylabel=r'$\Delta$Dec (deg)')
    
    return fig,     

every = 1
length = 10
frames = np.arange(0, base_period, every)    # iterable for the animation function. Chooses which frames (indices) to animate.
fps = len(frames) // length  # fps for the final animation

ani = animation.FuncAnimation(fig, animate, frames=frames, blit=True, repeat=False)
ani.save(f"folded_movie.mp4", writer='ffmpeg', fps=fps) 
ani.save(f"folded_movie.gif", writer='ffmpeg', fps=fps)
#pbar.close()
plt.close()
