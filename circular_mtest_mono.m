% load the data from the .mat file
% load('circular_data_panel_d_mono.mat', 'array_1', 'array_2', 'array_3', 'array_4');
% 
% array_1 = double(array_1);
% array_2 = double(array_2);
% array_3 = double(array_3);
% array_4 = double(array_4);
% 
% 
% % calculate the circular_mtest result for each array
% [pval1d, z] = circ_mtest(array_1, 0, 0.05);
% [pval2d, z] = circ_mtest(array_2, 0, 0.05);
% [pval3d, z] = circ_mtest(array_3, 0, 0.05);
% resultd4 = circ_mtest(array_4, 0, 0.05);


% load the data from the .mat file
load('circular_data_panel_e_mono.mat', 'array_1', 'array_2', 'array_3', 'array_4');

array_1 = double(array_1);
array_2 = double(array_2);
array_3 = double(array_3);
array_4 = double(array_4);

% calculate the circular_mtest result for each array
[pval1e, z] = circ_mtest(array_1, 0, 0.05);
[pval2e, z] = circ_mtest(array_2, 0, 0.05);
[pval3e, z] = circ_mtest(array_3, 0, 0.05);
[pval4e, z] = circ_mtest(array_4, 0, 0.05);