function [minx, maxx, miny, maxy] = imgBound(proj1, proj2)
%     Bound as [minx, maxx, miny, maxy]
minx = min([proj1(1, :), proj2(1, :)]);
miny = min([proj1(2, :), proj2(2, :)]);
maxx = max([proj1(1, :), proj2(1, :)]);
maxy = max([proj1(2, :), proj2(2, :)]);
% bound = [minx, maxx, miny, maxy];
end