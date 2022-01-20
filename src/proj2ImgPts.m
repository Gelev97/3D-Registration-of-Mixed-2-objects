function ret_val = proj2ImgPts(proj, minval, ratios, imgsize, boundratio)
    tablex = floor((proj(1, :) - minval(1)) / ratios(1) + imgsize(1) * boundratio);
    tabley = floor((proj(2, :) - minval(2)) / ratios(2) + imgsize(2) * boundratio);
    ret_val = [tablex; tabley];
end
