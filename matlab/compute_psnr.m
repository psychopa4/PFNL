function psnr=compute_psnr(img1,img2,scale)
if size(img1, 3) == 3,
    img1 = rgb2ycbcr(img1);
    img1 = img1(:, :, 1);
end

if size(img2, 3) == 3,
    img2 = rgb2ycbcr(img2);
    img2 = img2(:, :, 1);
end
boundarypixels = 0; 
img1 = img1(boundarypixels+1:end-boundarypixels,boundarypixels+1:end-boundarypixels,:);
img2 = img2(boundarypixels+1:end-boundarypixels,boundarypixels+1:end-boundarypixels,:);
imdff = double(img1) - double(img2);
imdff = imdff(:);

rmse = sqrt(mean(imdff.^2));
psnr=20*log10(255/rmse);