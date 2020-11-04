addpath('C:\Users\Lukas\Documents\projects\eeg_inverse_solutions\matlab\brainstorm3')
modelpath = 'C:\Users\Lukas\Documents\projects\eeg_inverse_solutions\matlab\results\model\';
sourcepath = 'C:\Users\Lukas\Documents\projects\eeg_inverse_solutions\matlab\results\sources';

protocol = 'Protocol02';
mni_positions = get_bst_vertice_positions(protocol);

save(strcat(modelpath, '\mni_positions.mat'), 'mni_positions', '-v7')

for i=0:9
    disp(i)
end
    