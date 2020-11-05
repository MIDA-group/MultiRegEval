function [registered_image]=BDcreation_reg(BDCparameters)
% function [registered_image]=BDcreation_reg(HE_image,SHG_image,pixelpermicron)

% November 3,2016: this function was originally developed by Adib Keikhosravi, a
% LOCI graduate student, to automatically registrate RGB bright field
% HE image with the correspoinding SHG image. This output registered image will
% be used for later segmentaion. The function is pushed to the github for a verison
% control.
% input:BDCparameters
% BDCparameters.HEfilepath: HE file path;
% BDCparameters.HEfilename: HE file name;
% BDCparameters.pixelpermicron: pixel per micron ratio of the HE image;
% BDCparameters.areaThreshold: threshold of the segmented areas; (for possible future use in segmentation)
% BDCparameters.SHGfilepath: SHG image file path ;
% Output: registered_image: registered,image

HEfilepath = BDCparameters.HEfilepath;
HEfilename = BDCparameters.HEfilename;
SHGfilename = HEfilename;
pixelpermicron = BDCparameters.pixelpermicron; % pixel size for HE image
SHGfilepath = BDCparameters.SHGfilepath;
HE_image = fullfile(HEfilepath,HEfilename);

%
SHGdata = fullfile(SHGfilepath,SHGfilename); 
fixedSHG  = imread(SHGdata);
fixedSHG=imadjust(fixedSHG); 

%
HEdata = imread(HE_image);
HEdata=im2double(HEdata);
max_HEdata=max(max(max(HEdata)));
HEdata_adj = imadjust(HEdata,[0 max_HEdata],[0 1]);
HEdata_adj=im2uint8(HEdata_adj);
HEdata_nuclei=HEdata_adj;
HEdata_red=HEdata_adj;
HEdata_gray=rgb2gray(HEdata_adj);
HEdata_gray=im2double(HEdata_gray);
HEdata_gray1=HEdata_gray; %%% ? not used?
HERGB = HEdata_adj;

S = decorrstretch(HERGB,'tol',0.01); %%% ? highlight color difference
HEdata_adj=S;

%%% image segmentation based on color -> nuclei, red, discard nuclei
[m,n]=size(HEdata_gray);
for i=1:m
    for j=1:n
        if(HEdata_adj(i,j,1)<120&&HEdata_adj(i,j,2)>150&&HEdata_adj(i,j,3)<120)
            HEdata_nuclei(i,j,:)=HEdata_adj(i,j,:);
        else
            HEdata_nuclei(i,j,:)=0;
        end
        if(HEdata_adj(i,j,1)>200&&HEdata_adj(i,j,2)<100&&HEdata_adj(i,j,3)>100)
            HEdata_red(i,j,1)=HEdata_adj(i,j,1);
        else
            HEdata_red(i,j,:)=0;
        end
    end
end

cform = makecform('srgb2lab');
lab_HEdata = applycform(HEdata_red,cform); %%% convert to lab colorspace why?
ab = double(lab_HEdata(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
% repeat the data clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
    'Replicates',3); %%% kmeans clustering
pixel_labels = reshape(cluster_idx,nrows,ncols);
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

%%% perform segmentation on original image
figure('position',[150 200 750 250],'NumberTitle','off','Name',HEfilename,'Visible', 'off');
 for k = 1:nColors
    color = HEdata;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
    tit=[ 'objects in cluster ' num2str(k)];
    subplot(1,nColors,k)
    imshow(segmented_images{k}), title(tit);
end

mean_cluster_value = mean(cluster_center,2);
[tmp, idx] = sort(mean_cluster_value);
HE_collagen=im2double(rgb2gray(segmented_images{idx(nColors)}));
gray_nuclei=im2double(rgb2gray(HEdata_nuclei));
h_nuclei = fspecial('gaussian',floor(pixelpermicron), 0.5);
nuclei_filtered = imfilter(im2double(gray_nuclei),h_nuclei);
BW_nuclei=im2bw(im2double(nuclei_filtered),0.001);
BW_nuclei_discard = bwareaopen(BW_nuclei, ceil(50*pixelpermicron^2));
se = strel('disk',floor(pixelpermicron));
BW_nuclei_dilated = imdilate(BW_nuclei_discard,se);
BW_nuclei_filled = imfill(BW_nuclei_dilated,'holes');
HE_collagen_exclude=HE_collagen.*(~BW_nuclei_filled);
HE_collagen_BW=im2bw(HE_collagen_exclude,0.01);
BW_discard = bwareaopen(HE_collagen_BW, ceil(pixelpermicron^2));
HE_collagen_exclude=HE_collagen_exclude.*BW_discard;


HEmoving=imresize(HE_collagen_exclude,size(fixedSHG));
[optimizer,metric] = imregconfig('multimodal');
disp(optimizer);
disp(metric);
optimizer.InitialRadius = optimizer.InitialRadius/3.5;
movingRegisteredAdjustedInitialRadius = imregister(HEmoving, fixedSHG, 'rigid', optimizer, metric);
optimizer.MaximumIterations = 1000;
%tformSimilarity = imregtform(HEmoving,fixedSHG,'similarity',optimizer,metric);
tformSimilarity = imregtform(HEmoving,fixedSHG,'rigid',optimizer,metric);
RfixedSHG = imref2d(size(fixedSHG));
tformSimilarity.T;
[movingRegisteredAffineWithIC treg tform]= imreg_new3(HEmoving,fixedSHG,'rigid',optimizer,metric,...
    'InitialTransformation',tformSimilarity);
HERmoving=imref2d(size(HEmoving));
HEdata_registered=imresize(HEdata,size(fixedSHG));
B = imwarp(HEdata_registered,HERmoving,tform,'OutputView',RfixedSHG,'FillValues',255);
% figure;imshowpair(B, fixedSHG); title('registered');
registered_image=B;

savePath = fullfile(HEfilepath,'HE_registered');
if ~exist(savePath,'dir')
    mkdir(savePath);
end
imwrite(registered_image,fullfile(savePath,HEfilename))
str111=[HEfilepath, 'tform_',HEfilename(1:end-3),'mat'];
% str111=[HEfilepath, 'test.mat'];
save(str111,'tform')

disp(sprintf('registered image %s was saved at %s',HEfilename,savePath))
close all;
    function [movingReg,Rreg,tform] = imreg_new3(varargin)
        %IMREGISTER Register two 2-D or 3-D images using intensity metric optimization.
        %
        %
        
        tform = imregtform(varargin{:});
        
        % Rely on imregtform to input parse and validate. If we were passed
        % spatially referenced input, use spatial referencing during resampling.
        % Otherwise, just use identity referencing objects for the fixed and
        % moving images.
        spatiallyReferencedInput = isa(varargin{2},'imref2d') && isa(varargin{4},'imref2d');
        if spatiallyReferencedInput
            moving  = varargin{1};
            Rmoving = varargin{2};
            Rfixed  = varargin{4};
        else
            moving = varargin{1};
            fixed = varargin{2};
            if (tform.Dimensionality == 2)
                Rmoving = imref2d(size(moving));
                Rfixed = imref2d(size(fixed));
            else
                Rmoving = imref3d(size(moving));
                Rfixed = imref3d(size(fixed));
            end
        end
        
        % Transform the moving image using the transform estimate from imregtform.
        % Use the 'OutputView' option to preserve the world limits and the
        % resolution of the fixed image when resampling the moving image.
        [movingReg,Rreg] = imwarp(moving,Rmoving,tform,'OutputView',Rfixed);
        
        
    end
end
