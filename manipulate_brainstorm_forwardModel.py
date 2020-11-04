import scipy.io
mat = scipy.io.loadmat('C:/Users/Lukas/Documents/projects/eeg_inverse_solutions/matlab/protocols/Protocol01/data/NewSubject/Default/headmodel_surf_openmeeg.mat')
print(f'loaded mat: {mat}')
print(f'type: {type(mat)}')