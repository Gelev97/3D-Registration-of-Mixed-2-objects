function pts = project3D(P, pts)
    pts = P * pts;
    pts = pts ./ pts(3, :);
end
