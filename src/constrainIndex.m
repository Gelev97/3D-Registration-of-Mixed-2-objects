function coor = constrainIndex(coor, imgsize)
    imgr = imgsize(1);
    imgc = imgsize(2);
    logits = coor(1, :) > 0 & coor(1, :) < imgr;
    logits = logits & (coor(2, :) > 0) & (coor(2, :) < imgc);
    coor = coor(:, logits); 
end
