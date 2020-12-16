pt1 = reshape(magic(6), [3, 12]);
pt1 = [pt1; ones(1, 12)];
gmat = eye(4);
gmat(1:3, 4) = [1;4;2];
gmat(1:3, 1:3) = rotz(60) * roty(24) * rotx(30);
pt1_tf = gmat * pt1;


pt1_noh = pt1(1:end-1, :);
[r, c] = size(pt1_noh);
l1 = [-pt1_noh', -ones(c, 1), zeros(c, 8), pt1_noh'.* pt1_tf(1, :)' , pt1_tf(1, :)'];
l2 = [zeros(c, 4), -pt1_noh', -ones(c, 1), zeros(c, 4), pt1_noh'.* pt1_tf(2, :)' , pt1_tf(2, :)'];
l3 = [zeros(c, 8), -pt1_noh', -ones(c, 1), pt1_noh'.* pt1_tf(3, :)' , pt1_tf(3, :)'];

A_mat = [l1;l2;l3];
[u, s, vt] = svd(A_mat);

v_mat = vt(:, end);
vmat = reshape(v_mat, [4,4])';

recovered = vmat * pt1;
recovered = recovered ./ recovered(4, :);

pt1_tf - recovered
