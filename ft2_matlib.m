clear all
close all
clc

im=imread("OriginalImage/1.jpg");
figure (1)
imshow(im);

im_if=fft2(im);
phase=angle(im_if)*255;

phase_im=uint8(phase);
phase_r=phase_im(:,:,1);
figure (2)
imshow(phase_im);

imwrite(phase_im, 'holo_mat.png');

