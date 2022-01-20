function T = rotMat(R, P, Y, t)
    [r, c] = size(t);
    assert(r == 3 && c == 1);
    T = rotz(Y)  * roty(P) * rotx(R);
    T = [T, t; 0,0,0,1];
end
