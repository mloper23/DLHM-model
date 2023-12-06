function ift = ifts(I)
    ift = ifftshift(ifft2(fftshift(I)));
end
