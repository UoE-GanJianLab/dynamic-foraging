% load the data from the .mat file
load('circular_data_panel_d_all.mat', 'array_1', 'array_2', 'array_3', 'array_4');

array_1 = double(array_1);
array_2 = double(array_2);
array_3 = double(array_3);
array_4 = double(array_4);


% calculate the circular_mtest result for each array
resultd1 = circ_mtest(array_1, 0, 0.05);
resultd2 = circ_mtest(array_2, 0, 0.05);
resultd3 = circ_mtest(array_3, 0, 0.05);
resultd4 = circ_mtest(array_4, 0, 0.05);


% load the data from the .mat file
load('circular_data_panel_e_all.mat', 'array_1', 'array_2', 'array_3', 'array_4');

array_1 = double(array_1);
array_2 = double(array_2);
array_3 = double(array_3);
array_4 = double(array_4);

% calculate the circular_mtest result for each array
resulte1 = circ_mtest(array_1, 0, 0.05);
resulte2 = circ_mtest(array_2, 0, 0.05);
resulte3 = circ_mtest(array_3, 0, 0.05);
resulte4 = circ_mtest(array_4, 0, 0.05);