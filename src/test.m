%% Setup
clc
clear

table_fname = "clear_table.txt";
table_M = readmatrix(table_fname, 'Delimiter', ' ');

hallway_fname = "clean_hallway.txt";
hallway_M = readmatrix(hallway_fname, 'Delimiter', ' ');


figure;
plot3(table_M(:, 1), table_M(:, 2), table_M(:, 3), 'g.');
hold on;
plot3(hallway_M(:, 1), hallway_M(:, 2), hallway_M(:, 3), 'b.');
axis equal;

%%

[r, ~] = size(table_M);
[r2, ~] = size(hallway_M);
clutered = [table_M', hallway_M'; ones(1, r + r2)];

fprintf('WARNING, The following operation generates ~3G of RAM usage, press enter to proceed\n');
pause;
% This operation generates 3GB of Ram on my desktop, proceed with caution
D = pdist2(clutered', clutered', 'euclidean');
assert(issymmetric(D));
extract_vectors = D([1,5,7, end, end-5, end-10], :);
clear D
%%
simple_displacement = [eye(3), [1;2;-1]; 0,0,0,1];
clutered(:, 1:r) = simple_displacement * clutered(:, 1:r);
figure;
plot3(clutered(1, 1:r), clutered(2, 1:r), clutered(3, 1:r), 'g.');
hold on;
plot3(clutered(1, r:end), clutered(2, r:end), clutered(3, r:end), 'b.');
axis equal;

D = pdist2(clutered', clutered', 'euclidean');
assert(issymmetric(D));
extract_vectors_2 = D([1,5,7, end, end-5, end-10], :);
