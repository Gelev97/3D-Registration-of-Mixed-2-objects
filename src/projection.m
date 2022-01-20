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
[table_r, ~] = size(table_M);
[hw_r, ~] = size(hallway_M);

homo_table = [table_M, ones(table_r, 1)]';
homo_hw = [hallway_M, ones(hw_r, 1)]';

K = eye(3);
w2K = [eye(3), zeros(3,1)];
P = K * w2K;
% P = [1, 0, 0, 0;
%      0, 1, 0, 0;
%      0, 0, 1, 0];

proj_table = project3D(P, homo_table);
proj_hw = project3D(P, homo_hw);


%% Initial Figure
imgr = 1000;
imgc = 1000;
boundary_ratio = 0.01;

img = zeros(imgr, imgc);
minx = -3;
maxx = 3;
miny = -3;
maxy = 3;

x_range = (maxx - minx);
y_range = (maxy - miny);
ratios = [x_range / imgr, y_range / imgc] / (1 - 2 * boundary_ratio);

img_table = proj2ImgPts(proj_table, [minx, miny], ratios, [imgr, imgc], boundary_ratio);
img_hw = proj2ImgPts(proj_hw, [minx, miny], ratios, [imgr, imgc], boundary_ratio);

linear_table_idx = sub2ind([imgr, imgc], img_table(1, :), img_table(2, :));
linear_hw_idx = sub2ind([imgr, imgc], img_hw(1, :), img_hw(2, :));

img(linear_table_idx) = 255;
img(linear_hw_idx) = 255;

subplot(1,2,1);
% figure;
image(img);
% axis equal;

%% Final Figure

g1 = eye(4);
g1(1:3, 1:3) = rotx(20);
g1(1:3, 4) = [-6/7 ; 0.8; -1.3];

g2 = eye(4);
g2(1:3, 1:3) = rotz(20);
g2(1:3, 4) = [-1.4; 0.6; 1.2];

% Get Transformed points
tf_table = g1 * homo_table;
% Project Points to Image Plane with the same P
tfproj_table = project3D(P, tf_table);

% Repeat
tf_hw = g2 * homo_hw;
tfproj_hw = project3D(P, tf_hw);

% Transform points to Image indices
tfimg_table = proj2ImgPts(tfproj_table, [minx, miny], ratios, [imgr, imgc], boundary_ratio);
tfimg_hw = proj2ImgPts(tfproj_hw, [minx, miny], ratios, [imgr, imgc], boundary_ratio);

tfimg_table = constrainIndex(tfimg_table, [imgr, imgc]);
tfimg_hw = constrainIndex(tfimg_hw, [imgr, imgc]);

% Transform 2D indices to linear indices
tflinear_table_idx = sub2ind([imgr, imgc], tfimg_table(1, :), tfimg_table(2, :));
tflinear_hw_idx = sub2ind([imgr, imgc], tfimg_hw (1, :), tfimg_hw (2, :));

img2 = zeros(imgr, imgc);
img2(tflinear_table_idx) = 255;
img2(tflinear_hw_idx) = 255;

subplot(1,2,2);
% figure;
image(img2);
% axis equal;
