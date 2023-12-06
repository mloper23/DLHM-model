
function ft = fts(I)
    ft = ifftshift(fft2(fftshift(I)));
end