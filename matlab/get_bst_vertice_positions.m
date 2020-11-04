function mni_positions = get_bst_vertice_positions(protocol)
    path_T1 = strcat('C:\Users\Lukas\Documents\projects\eeg_inverse_solutions\matlab\protocols\', protocol, '\anat\@default_subject\subjectimage_T1.mat');
    path_sourcemodel = strcat('C:\Users\Lukas\Documents\projects\eeg_inverse_solutions\matlab\protocols\', protocol, '\anat\@default_subject\tess_cortex_pial_low_5124V.mat');
    T1 = load(path_T1);
    sourcemodel = load(path_sourcemodel);
    % Get mni positions in mm (thus times 1000)
    mni_positions = cs_convert(T1  , 'scs', 'mni', sourcemodel.Vertices) * 1000;
end