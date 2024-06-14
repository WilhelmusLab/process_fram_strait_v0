clear all
close all

% HOME='/rhome/rlope040/rotation/rot2003';                                %server
% HOME='/rhome/rlope040/dispersion/disp2003';
HOME='/rhome/rlope040/single_dispersion/disp2010';%server

M_WORKERS=32;
%change these depending on the size of your image for equal spacing into
%tiles for image processing, such that size(im,1)/X=A1 and size(im,2)/X=A2
A1=710;   
A2=848;
        
% for NE Greenland, use these values
%A1=1015;
%A2=743;
        

%select year
BEGINNING=734227;          %pick day for first image
datedisp(BEGINNING+1);     %test
%2016=736420, 2017=736785, 2018=737150
ROWS=30000;                %estimated number of floes for the season



%% LANDMASK
%From NASA's masked image, obtain landmass information
cd(strcat(HOME,'/input/info')) 
land = imread('Land.tif');

land1=land(:,:,1);
land2=land(:,:,2);
land3=land(:,:,3);

LAND=find(land1~=0);

%apply mask to all images
cd(strcat(HOME,'/input/images')) 
im_files = dir('*.tif');

for i=1:size(im_files,1)
cd(strcat(HOME,'/input/images')) 
    
    fnm = im_files(i).name;
    im = imread(fnm);
    info_region= geotiffinfo(fnm);
    pixels_region = imread(fnm);
    
    Red_region = pixels_region(:,:,1);
    Green_region = pixels_region(:,:,2);
    Blue_region = pixels_region(:,:,3);
    
    doublered_region = im2double(Red_region);
    doublegreen_region = im2double(Green_region);
    doubleblue_region = im2double(Blue_region);
    
    doublered_region(LAND) = NaN;
    doublegreen_region(LAND) = NaN;
    doubleblue_region(LAND) = NaN;
        
    rgbImage = cat(3, doublered_region, doublegreen_region, doubleblue_region);
    
    % save images
    cd(strcat(HOME,'/output/masked')) 
    imwrite(rgbImage,fullfile(fnm),'tiff', 'Compression','none')  

end


%% ORDER OF IMAGES
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file A1 A2
close all

cd(strcat(HOME,'/input/info'))
[~,aqua_pass]  = xlsread('sat_2010.xlsx','A1:A173');
[~,terra_pass]  = xlsread('sat_2010.xlsx','B1:B173');
index=1;
for i=1:size(aqua_pass,1)
    if datetime(aqua_pass{i,1}) > datetime(terra_pass{i,1})
    order(index) = index+1;
    order(index+1) = index;
    time_stamp{index,1} = terra_pass{i,1};
    time_stamp{index+1,1} = aqua_pass{i,1};
    
    else
    order(index) = index;
    order(index+1) = index+1;
    time_stamp{index,1} = aqua_pass{i,1};
    time_stamp{index+1,1} = terra_pass{i,1};    
    end
    index=index+2;
end



for i=1:size(order,2)-1
    str2=time_stamp{i+1,1};
    str1=time_stamp{i,1};
    t2 = datevec(str2,'mmmm dd, yyyy HH:MM:SS');
    t1 = datevec(str1,'mmmm dd, yyyy HH:MM:SS');
    delta_t(i) = etime(t2,t1)/60;
end 

delta_t=delta_t';

cd(strcat(HOME,'/output/tracked'))
save(['delta_t'],'delta_t');
save(['order'],'order');


cd(strcat(HOME,'/output/masked'))
im_files = dir('*.tif');

INDEX=0;
for i = order
    INDEX=INDEX+1;
    cd(strcat(HOME,'/output/masked'))
    fnm = im_files(i).name;
    if contains(fnm,'aqua')
    sat_order(INDEX,1)=0;
    elseif contains(fnm,'terra')
    sat_order(INDEX,1)=1;
    end
    
end

cd(strcat(HOME,'/output/tracked'))
save(['sat_order'],'sat_order');


%% IMAGE PROCESSING
%finds automatic gamma and imadjust to adjust intensity of image 
%imfill grey image + gamma selection+ imadjust + adapthresh + imbinarize 
%+ imfill BW only away from the coast + segmentation. Additionally stores 
%region properties of every ice floe 
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file A1 A2
close all

%% make coast images
cd(strcat(HOME,'/input/info')) %get image info

pixels = imread('Land.tif');
pixel=pixels(:,:,1);
bw = zeros(size(pixel));

filled_coastline=bw;
filled_coastline(LAND)=1;
%imshow(filled_coastline)

filled_coastline=logical(filled_coastline);

%black
cd(strcat(HOME,'/input/info'))
imwrite(filled_coastline,'black_raw.tif','tiff', 'Compression','none')  

%white
filled_coastline2 = imcomplement(filled_coastline);
imwrite(filled_coastline2,'white_raw.tif','tiff', 'Compression','none')  

%grey
im1 = double(filled_coastline);

red = im1; %red channel
green = im1; %blue channel
blue = im1; %greenchannel

%turns landmass grey
red(filled_coastline==1)=.75;
green(filled_coastline==1)=.75;
blue(filled_coastline==1)=.75;

BW=cat(3,red,green,blue);
%figure,imshow(BW)
imwrite(BW,'grey_raw.tif','tiff', 'Compression','none')  


%thicker coastline
im_coast = filled_coastline; 
im_coast2 = bwperim(im_coast,8);
im_coast3 = imdilate(im_coast2, strel('disk',50));  %contours border to get rid of soft ice

filled = imfill(im_coast3,'holes');



%% axis data
cd(strcat(HOME,'/input/info'))
info_region= geotiffinfo('Land.tif'); 
[a,b,~]=size(imread('Land.tif'));

info_region.BoundingBox     %this reads:[left top corner, right top corner, right 
                            %bottom corner, left bottom corner] before I divided by 1000 now 1
x_region=[info_region.BoundingBox(1)/1:info_region.RefMatrix(2)/1:info_region.BoundingBox(2)/1];  
y_region=[info_region.BoundingBox(4)/1:info_region.RefMatrix(4)/1:info_region.BoundingBox(3)/1];
 
y1_region=(y_region(1:end-1)+y_region(2:end))/2;
x1_region=(x_region(1:end-1)+x_region(2:end))/2;
 
[X_region,Y_region]=meshgrid(x1_region,y1_region);

A=X_region(size(X_region,1),1);
B=X_region(size(X_region,1),size(X_region,2));
C=Y_region(1,1);
D=Y_region(size(Y_region,1),1);


%decide where and what values for your plot ticks
list111= linspace(A,B,5);
list222= linspace(C,D,5);

exp1=real(floor(log10(list111(1))));
exp2=real(floor(log10(list222(1))));

exp11=exp1-3;
EXP1 = strcat('10^{', num2str(exp11), '}');

exp22=exp2-3;
EXP2 = strcat('10^{', num2str(exp22), '}');

list11= linspace(A,B,5)/10^(exp1);
list22= linspace(C,D,5)/10^(exp2);

difference1=abs(diff(list11)); 
difference2=abs(diff(list22));

diff1=difference1(1)*4;
list111= 0:difference1(1):diff1;

diff2=difference2(1)*4;
list222= 0:difference2(1):diff2;
LIST1={};
LIST2={};

for f=1:size(list11,2)
LIST1{f} = sprintf ('%.1f', list111(f));
LIST2{f} = sprintf ('%.1f', list222(f));
end

LIST2=fliplr(LIST2);


%% begin to process slices of image
cd(strcat(HOME,'/output/tracked'))
order = importdata('order.mat');  

cd(strcat(HOME,'/input/images'))
im_files = dir('*.tif');

%title date
for i=1:size(im_files,1)/2
date_title{1,i}=datedisp([BEGINNING+i]);
end
date_title = repelem(date_title,2);

value=find(filled==1);

col_num1=repmat(1:size(pixels,2)/A2,size(pixels,1)/A1);
col_num=col_num1(1,:)';

row_num1=repmat(1:size(pixels,1)/A1,size(pixels,2)/A2);
row_num2=row_num1(:)';
row_num=row_num2(1:size(pixels,1)/A1*size(pixels,2)/A2)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parfor (i1=1:size(order,2),M_WORKERS) 
i=[];
nlabel=[];
properties={};
labels=[];
image_final=[];
properties1=[];
properties2=[];
prop=[];

i=i1;
%load RGB image    
cd(strcat(HOME,'/input/images'))
fnm = im_files(i).name;
im = imread(fnm);
[a,b,~]=size(im);

% figure,imshow(im);
% title('original');


%crop  original image to area of interest
im2 = im; 

%original RGB image 
%%figure,imshow(im2)
im2a=im2(:,:,1);
im2b=im2(:,:,2);
im2c=im2(:,:,3);


%load reflectance image
cd(strcat(HOME,'/input/images_reflectance'))
fnm = im_files(i).name;
ref_im = imread(fnm);
%crop  original image to area of interest
ref_im2 = ref_im; 

%original reflectance image
%%figure,imshow(ref_im2)
ref_im2a=ref_im2(:,:,1);
ref_im2b=ref_im2(:,:,2);
ref_im2c=ref_im2(:,:,3);

%clouds are mutted in reflectance image
ref_im2aa=zeros(size(ref_im2a));
ref_im3=cat(3,ref_im2aa,ref_im2b,ref_im2c);
%%figure,imshow(ref_im3)       %%%%          

ref_im3a=ref_im3(:,:,1);
ref_im3b=ref_im3(:,:,2);
ref_im3c=ref_im3(:,:,3);

%% identify cloud-ice, then deletes on clouds(ref_im2a) what is not cloud-ice
ref_im4=ref_im2;
ref_im4a=ref_im2(:,:,1);
ref_im4b=ref_im2(:,:,2);
ref_im4c=ref_im2(:,:,3);
% figure,imshow(ref_im4a)
% figure,imshow(ref_im4b)
% figure,imshow(ref_im4c)

%identify cloud-ice
mask_aa = (ref_im4a < 200);  %cloud to ice ratio ch1
mask_bb = (ref_im4b > 190);  %cloud to ice ratio ch2
%figure,imshow(mask_aa)
%figure,imshow(mask_bb)

F_ci=find(mask_aa==1 & mask_bb==1);

L_ci=[];
L_ci(:,1)=1:(a*b);
LL2=(find(~ismember(L_ci,F_ci)==1));

ref_im5=ref_im4;
ref_im5a = ref_im4(:,:,1); 
ref_im5b = ref_im4(:,:,2);
ref_im5c = ref_im4(:,:,3); 

ref_im5a(LL2)=0;
ref_im5b(LL2)=0;

%only these fall into the correct cloud-ice ratio
cloud_ice=double(ref_im5a)./double(ref_im5b);
mask_cloud_ice = (cloud_ice >= 0 & cloud_ice < .75);  %this is ice-cloud
%figure,imshow(mask_cloud_ice)
A_ci2=find(mask_cloud_ice==1);

%delete from ref2a above 110
ref_imclouds=(ref_im2a>110);
%figure,imshow(ref_imclouds)

ref_imclouds(A_ci2)=0;
%figure,imshow(ref_imclouds)        %these are clouds, delete from originals

A_ci3=find(ref_imclouds==1);

ref_im2a(A_ci3)=0;
ref_im2b(A_ci3)=0;
ref_im2c(A_ci3)=0;
ref_im6=cat(3,ref_im2a,ref_im2b,ref_im2c);
%figure,imshow(ref_im6)       %%%%                       %delete these clouds

% 
% figure,imshow(ref_im2)       %%%%                       %delete these clouds
% figure,imshow(ref_im2b)       %%%%                       %delete these clouds
% figure,imshow(ref_im2c)       %%%%                       %delete these clouds

ref_im7=ref_im6;
ref_im7(:,:,1)=zeros;    %mute clouds

ref_im6a = ref_im6(:,:,1); 
ref_im6b = ref_im6(:,:,2);
ref_im6c = ref_im6(:,:,3);

%figure,imshow(ref_im6a) 
%figure,imshow(ref_im6b) 
%figure,imshow(ref_im6c) 


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd(strcat(HOME,'/output/cloud1'))
imwrite(ref_im6a,fullfile(fnm),'tiff', 'Compression','none') 

cd(strcat(HOME,'/output/cloud2'))
imwrite(ref_im6b,fullfile(fnm),'tiff', 'Compression','none') 

cd(strcat(HOME,'/output/cloud3'))
imwrite(ref_im6c,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

im3=im2;
% im3(ref_im6a~=0)=0;
%figure,imshow(im3)
%?nonlinear diffusion filtering
im4 = imdiffusefilt(im3,'NumberOfIterations',3);
%figure,imshow(im4)

%im5=im4; 
%% equalize RGB histograms in each channel
red  = im4(:,:,1);                   %red channel
green= im4(:,:,2);                 %blue channel
blue = im4(:,:,3);                  %green channel


clouds_ref=ref_im2a;
clouds_ref(value)=0;

cd(strcat(HOME,'/input/info'))
tiles_clouds=mat2tiles(clouds_ref,[A1 A2]);
tiles_red=mat2tiles(red,[A1 A2]);
tiles_green=mat2tiles(green,[A1 A2]);
tiles_blue=mat2tiles(blue,[A1 A2]);


gammared=[];
gammagreen=[];
gammablue=[];
for ig1=1:size(pixels,1)/A1*size(pixels,2)/A2
ig=row_num(ig1);
igg=col_num(ig1);
row1=(ig*A1)-A1+1:ig*A1;
col1=(igg*A2)-A2+1:igg*A2;        

    entrop1 = entropy(tiles_clouds{ig,igg});
    white1=sum(sum(tiles_clouds{ig,igg}>25.5));
    black1=sum(sum(tiles_clouds{ig,igg}<=25.5));
    if entrop1>4 && white1/(white1+black1)>.4
        
        
    gammared(row1,col1)=adapthisteq(tiles_red{ig,igg},'Range','original','NBins', 255);                %equalizes the histogram for the red channel
    gammagreen(row1,col1)=adapthisteq(tiles_green{ig,igg},'Range','original','NBins', 255);            %equalizes the histogram for the green channel
    gammablue(row1,col1)=adapthisteq(tiles_blue{ig,igg},'Range','original','NBins', 255);               %equalizes the histogram for the blue channel  

    else
    gammared(row1,col1)=tiles_red{ig,igg};
    gammagreen(row1,col1)=tiles_green{ig,igg};
    gammablue(row1,col1)=tiles_blue{ig,igg};
    end
end


gammared = uint8(255 * mat2gray(gammared));
gammagreen = uint8(255 * mat2gray(gammagreen));
gammablue = uint8(255 * mat2gray(gammablue));


im5=cat(3,gammared,gammagreen,gammablue); %concatenates the 3 channels to get rgb







im6=im5;
%figure,imshow(im6)

im6a=im6(:,:,1);
im6b=im6(:,:,2);
im6c=im6(:,:,3);



im6=cat(3,im6a,im6b,im6c); %concatenates the 3 channels to get rgb
%figure,imshow(im6)
% close all

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd(strcat(HOME,'/output/adapt_hist'))
imwrite(im6,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


% % convert RGB image to grayscale image
im9 = rgb2gray(im6);
im9(ref_im6a==0 & ref_im6b==0 & ref_im6c==0)=0;
%figure, imshow(im9)


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% cd(strcat(HOME,'/output/BW0'))
% imwrite(im9,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


%sharpen image
im7 = imsharpen(im9,'Radius',10,'Amount',2);  
%figure, imshow(im9)
%separate with filters
se = strel('disk',1);
Iobrd1 = imdilate(im7, se);
%%figure,imshow(Iobrd)
im8 = imreconstruct(imcomplement(Iobrd1), imcomplement(im7));
%figure,imshow(Iobrcbr) %%this one minus original
im8(value)=0;
%figure,imshow(im8)




%%%%%%%%%%%%%%%%%%%
%cover landmass
I = im9;
I(value)=0;
%%figure,imshow(I)

%separate with filters
se = strel('disk',1);
Iobrd = imdilate(I, se);
%%figure,imshow(Iobrd)
Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(I));
%figure,imshow(Iobrcbr) %%this one minus original
Iobrcbr(value)=0;
%figure,imshow(Iobrcbr)


brighten=Iobrcbr-im6b;
%figure,imshow(brighten)


%%%%% other option to make 1 image 
im9(value)=0;
im99=im2double(im9);
%figure,imshow(im99)

D111=im99;

D111(brighten~=0)=D111(brighten~=0)*.1;
%figure,imshow(D111)
D1111 = uint8(255 * mat2gray(D111));

new1=imsubtract(D1111,(Iobrcbr));
%figure,imshow(imcomplement(new1))




%figure,imshow(imcomplement(imadjust(im8,[],[],1.5)))
%figure,imshow(imadjust(new1,[],[],0.5))

im10=imcomplement(imadjust(im8,[],[],1.5));
new1(im10>220)=new1(im10>220)*1.3;
%figure,imshow(new1)



%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% cd(strcat(HOME,'/output/BW1'))
% imwrite(new1,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%sharpen image
% im10(new1<=.30)=0;
% figure,imshow(im10)





ref_im2aa=ref_im2(:,:,1);
ref_im2bb=ref_im2(:,:,2);
ref_im2cc=ref_im2(:,:,3);

ref_im2aa(value)=0;
ref_im2bb(value)=0;
ref_im2cc(value)=0;









cd(strcat(HOME,'/input/info'))
tiles_new1=mat2tiles(new1,[A1 A2]);
tiles_refa=mat2tiles(ref_im2aa,[A1 A2]);
tiles_refb=mat2tiles(ref_im2bb,[A1 A2]);
tiles_refc=mat2tiles(ref_im2cc,[A1 A2]);


bw1=[];
bw2=[];
for ig1=1:size(pixels,1)/A1*size(pixels,2)/A2
ig=row_num(ig1);
igg=col_num(ig1);
row1=(ig*A1)-A1+1:ig*A1;
col1=(igg*A2)-A2+1:igg*A2;        
    
    bw2(row1,col1) = imbinarize(tiles_new1{ig,igg},'adaptive','ForegroundPolarity','dark');
       
        
    ab = tiles_new1{ig,igg};
    ab = im2single(ab);
    tiles_new2 = imsegkmeans(ab,3,'NumAttempts',4); %repeat clustering x4 to avoid local minima
%     figure,imshow(tiles_new2,[])
        
    tiles_ref1a=tiles_refa{ig,igg};
    tiles_ref1b=tiles_refb{ig,igg};
    tiles_ref1c=tiles_refc{ig,igg};
        
    
    
    
    
%find the label of the pixels are ice for sure ([0,230,240])
ice_a=find(tiles_ref1a<5);
ice_b=find(tiles_ref1b>230);
ice_c=find(tiles_ref1c>240);
ice_labels = intersect(intersect(ice_a,ice_b),ice_c);



%if there are no ice floes, relax floe parameters
if isempty(ice_labels)
ice_a=find(tiles_ref1a<10);
ice_c=find(tiles_ref1c>190);
ice_labels = intersect(intersect(ice_a,ice_b),ice_c);
    if isempty(ice_labels)
    ref22=tiles_ref1b;
    ref33=tiles_ref1c;
    ref22(ref22<75)=0;
    ref33(ref33<75)=0;
    yy1=imhist(ref22);
    [~,peakLocs1]=findpeaks(yy1,'sortstr','descend');
    yy1=imhist(ref33);
    [~,peakLocs2]=findpeaks(yy1,'sortstr','descend');
        if size(peakLocs1,1)>2 && size(peakLocs2,1)>2 
        peak1=peakLocs1(2);
        peak2=peakLocs2(2);
        ice_b=find(tiles_ref1b>peak1);
        ice_c=find(tiles_ref1c>peak2);
        ice_labels = intersect(intersect(ice_a,ice_b),ice_c);
        nlabel=mode(double(tiles_new2(ice_labels)));
        else
        nlabel=1;    
        end
    else  
    nlabel=mode(double(tiles_new2(ice_labels)));
    end
else
    nlabel=mode(double(tiles_new2(ice_labels)));
end

if isempty(nlabel) || isnan(nlabel)
   ice_b=find(tiles_ref1b>230);
   nlabel=mode(double(tiles_new2(ice_b)));
end 




labels=double(nlabel);
bw1(row1,col1)=(tiles_new2==labels);
end





%figure,imshow(bw1)
%figure,imshow(bw2)


%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd(strcat(HOME,'/output/BW1'))
imwrite(bw1,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd(strcat(HOME,'/output/BW2'))
imwrite(bw2,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!






bw1=logical(bw1);
bw2=logical(bw2);



BW1=bw1;
BW2=bw2;
BW_test = bwareaopen(bw1,20);
%figure,imshow(BW_test)

BW_test1 = bwmorph(BW_test,'hbreak');
%figure,imshow(BW_test1)

BW_test2 = bwmorph(BW_test1,'branch');
% %figure,imshow(BW_test2)

BW_test22 = bwmorph(BW_test2,'bridge');
%  %figure,imshow(BW_test22)


se1 = strel('disk', 4);
BW_test22 = imerode(BW_test22, se1);
%figure,imshow(BW5)

BW_test22= imdilate(BW_test22,se1);
%figure,imshow(BW6)

BW_test3=imfill(BW_test22,'holes');
%figure,imshow(BW_test3)

holes=find(BW_test3~=BW_test1);
BW1(holes)=1;
%figure,imshow(BW1)

segment1=zeros(a,b);
seg = -bwdist(~BW1);
mask2 = imextendedmin(seg,2);
seg2 = imimposemin(seg,mask2);
lab1 = watershed(seg2);
segment1(lab1 == 0) = 1;
%figure,imshow(segment1) 


%figure,imshow(BW1)
BW1(lab1 == 0)=0;

%figure,imshow(BW1)





BW_test = bwareaopen(bw2,20);
%figure,imshow(BW_test)

BW_test1 = bwmorph(BW_test,'hbreak');
%figure,imshow(BW_test1)

BW_test2 = bwmorph(BW_test1,'branch');
% %figure,imshow(BW_test2)

BW_test22 = bwmorph(BW_test2,'bridge');
%  %figure,imshow(BW_test22)


se1 = strel('disk', 4);
BW_test22 = imerode(BW_test22, se1);
%figure,imshow(BW5)

BW_test22= imdilate(BW_test22,se1);
%figure,imshow(BW6)

BW_test3=imfill(BW_test22,'holes');
%figure,imshow(BW_test3)

holes=find(BW_test3~=BW_test1);
BW2(holes)=1;
%figure,imshow(BW1)

segment2=zeros(a,b);
seg = -bwdist(~BW2);
mask2 = imextendedmin(seg,2);
seg2 = imimposemin(seg,mask2);
lab2 = watershed(seg2);
segment2(lab2 == 0) = 1;
%figure,imshow(segment1) 

lab3=zeros(size(lab2));
lab3(segment2==1 & segment1==1)=1;

BW11=bw1;
gmag = imgradient(histeq(new1));
%figure,imshow(gmag,[])

segment3=find(segment1==1 & segment2==1);



I=new1;
se = strel('disk',20);
Io = imopen(I,se);
% imshow(Io)
title('Opening')
Ie = imerode(I,se);
Iobr = imreconstruct(Ie,I);
% imshow(Iobr)
title('Opening-by-Reconstruction')
Ioc = imclose(Io,se);
% imshow(Ioc)
title('Opening-Closing')
Iobrd = imdilate(Iobr,se);
Iobrcbr = imreconstruct(imcomplement(Iobrd),imcomplement(Iobr));
Iobrcbr = imcomplement(Iobrcbr);
% imshow(Iobrcbr)
title('Opening-Closing by Reconstruction')
fgm = imregionalmax(Iobrcbr);
%figure,imshow(fgm)

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% cd(strcat(HOME,'/output/BW4'))
% imwrite(fgm,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

BW3=imadd(fgm,BW1);
%figure,imshow(BW3)

% I(fgm==1)=255;
% figure,imshow(I)

gmag2 = imimposemin(gmag, lab3 | BW3);

L = watershed(gmag2);

labels1=zeros(size(L));
labels1(L==0)=1;


%figure,imshow(cat(3,labels1,bw1,ones(size(bw1))))
%figure,imshow(cat(3,lab3,bw1,ones(size(bw1))))
all1=cat(3,lab3,labels1,bw1);
%figure,imshow(cat(3,lab3,labels1,bw1))

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd(strcat(HOME,'/output/BW3'))
imwrite(all1,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

fgm = uint8(255 * mat2gray(fgm));
new2=imadd(new1,fgm*.3);
%figure,imshow(new2)

new2(lab3==1)=0;
new2(labels1==1)=0;
%figure,imshow(new2)

new2=imfill(new2,'holes');


%sharpen image
im99 = imsharpen(new2,'Radius',10,'Amount',2);  
% im99(labels1==1 & bw1==0 & bw2==0)=0;
im99(labels1==1)=0;
%figure, imshow(im99)



%separate with filters
se = strel('disk',1);
Iobrd2 = imdilate(im99, se);
%%figure,imshow(Iobrd)
new3 = imreconstruct((Iobrd2), (im99));
%figure,imshow(Iobrcbr) %%this one minus original
new3(lab3==1)=0;
new3=imadd(new3,fgm*.5);


%figure,imshow(new3)

%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% cd(strcat(HOME,'/output/BW6'))
% imwrite(new3,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

cd(strcat(HOME,'/input/info'))
tiles_new1=mat2tiles(new3,[A1 A2]);
tiles_refa=mat2tiles(ref_im2aa,[A1 A2]);
tiles_refb=mat2tiles(ref_im2bb,[A1 A2]);
tiles_refc=mat2tiles(ref_im2cc,[A1 A2]);


for ig1=1:size(pixels,1)/A1*size(pixels,2)/A2
ig=row_num(ig1);
igg=col_num(ig1);
row1=(ig*A1)-A1+1:ig*A1;
col1=(igg*A2)-A2+1:igg*A2;   
    
    bw2(row1,col1)=imbinarize(tiles_new1{ig,igg},'adaptive','ForegroundPolarity','dark');
       
    ab = tiles_new1{ig,igg};
    ab = im2single(ab);
    tiles_new2 = imsegkmeans(ab,3,'NumAttempts',4); %repeat clustering x4 to avoid local minima
%     figure,imshow(tiles_new2,[])
        
    tiles_ref1a=tiles_refa{ig,igg};
    tiles_ref1b=tiles_refb{ig,igg};
    tiles_ref1c=tiles_refc{ig,igg};
        
    
    
    
%find the label of the pixels are ice for sure ([0,230,240])
ice_a=find(tiles_ref1a<5);
ice_b=find(tiles_ref1b>230);
ice_c=find(tiles_ref1c>240);
ice_labels = intersect(intersect(ice_a,ice_b),ice_c);



%if there are no ice floes, relax floe parameters
if isempty(ice_labels)
ice_a=find(tiles_ref1a<10);
ice_c=find(tiles_ref1c>190);
ice_labels = intersect(intersect(ice_a,ice_b),ice_c);
    if isempty(ice_labels)
    ref22=tiles_ref1b;
    ref33=tiles_ref1c;
    ref22(ref22<75)=0;
    ref33(ref33<75)=0;
    yy1=imhist(ref22);
    [~,peakLocs1]=findpeaks(yy1,'sortstr','descend');
    yy1=imhist(ref33);
    [~,peakLocs2]=findpeaks(yy1,'sortstr','descend');
        if size(peakLocs1,1)>2 && size(peakLocs2,1)>2 
        peak1=peakLocs1(2);
        peak2=peakLocs2(2);
        ice_b=find(tiles_ref1b>peak1);
        ice_c=find(tiles_ref1c>peak2);
        ice_labels = intersect(intersect(ice_a,ice_b),ice_c);
        nlabel=mode(double(tiles_new2(ice_labels)));
        else
        nlabel=1;    
        end
    else  
    nlabel=mode(double(tiles_new2(ice_labels)));
    end
else
    nlabel=mode(double(tiles_new2(ice_labels)));
end

if isempty(nlabel) || isnan(nlabel)
   ice_b=find(tiles_ref1b>230);
   nlabel=mode(double(tiles_new2(ice_b)));
end 




labels=double(nlabel);

tiles_new3=(tiles_new2==labels);
    
bw1(row1,col1)=(tiles_new2==labels); 
end
%figure,imshow(bw1)
%figure,imshow(bw2)


 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
cd(strcat(HOME,'/output/BW4'))
imwrite(bw1,fullfile(fnm),'tiff', 'Compression','none') 
%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




BW_final1=bwmorph(bw1,'hbreak');
%figure,imshow(BW_final1)


BW_final1(labels==1)=0;
%figure,imshow(BW_final1)

% BW_final2=bwmorph(BW_final1,'branch');
%figure,imshow(BW_final2)

% BW_final2(segment3)=0;
%figure,imshow(BW_final2)

BW_final3=bwmorph(BW_final1,'fill');
%figure,imshow(BW_final3)

BW_final3(labels==1 & lab3 ==1)=0;
%figure,imshow(BW_final3)


BW_final4=imfill(BW_final3,'holes');
%figure,imshow(BW_final4)  



BW4 = bwmorph(BW_final4,'branch');
%figure, imshow(BW4);


se1 = strel('disk', 1);
BW5 = imerode(BW4, se1);
%figure,imshow(BW5)

se2 = strel('disk', 2);
BW6= imdilate(BW5,se2);


BW6(labels==1)=0;


%figure,imshow(BW6)

BW7 = imreconstruct(BW4, BW6);
% figure,imshow(BW7)





cd(strcat(HOME,'/output/BW5'))
imwrite(BW7,fullfile(fnm),'tiff', 'Compression','none') 

% cd(strcat(HOME,'/input/info'))
% mask_bw=imread('mask.tif');

BW_binary=BW7;
% BW_binary(mask_bw==0)=0;



cd(strcat(HOME,'/output/BW'))
imwrite(BW_binary,fullfile(fnm),'tiff', 'Compression','none') 














%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BW4=binary;
BW_new=BW_binary;

BW_double=BW_new;

BW_new(find(filled==1))=0;
BW_double=double(BW_double);
red = BW_double; %red channel
green = BW_double; %blue channel
blue = BW_double; %greenchannel

%turns landmass grey
red(filled_coastline==1)=.75;
green(filled_coastline==1)=.75;
blue(filled_coastline==1)=.75;

BW_grey=cat(3,red,green,blue);
%figure,imshow(BW_grey)

%% obtain centroid
% properties matrix
% figure, imshow(BW_grey), hold on
[B,~] = bwboundaries(BW_new,'noholes');
stats = regionprops(BW_new,'Area','Centroid','MajorAxisLength','MinorAxisLength',...
    'Orientation','Perimeter', 'BoundingBox', 'ConvexArea','Solidity');
[clust,numb] = bwlabel(BW_new, 8);
properties1 = zeros(numb, 13);
if numb~=0
%% set thresholds for centroid output:
threshold_min = 100;     %few= 5000
threshold_max = 90000;     %few=15000
%% for loop over all images and plot centroids
%stores: area, perimeter, major axis, minor axis, orientation, centroidx,
%centroidy in one matrix
for m1 = 1:numb
    area = stats(m1).Area;
    if area>threshold_min && area<threshold_max
%         plot(centroids(1),centroids(2),'ko', 'MarkerFaceColor','r','MarkerSize',6);   %'MarkerSize',8
        properties1(m1,1)=stats(m1).Area;
        properties1(m1,2)=stats(m1).Perimeter;
        properties1(m1,3)=stats(m1).MajorAxisLength;
        properties1(m1,4)=stats(m1).MinorAxisLength;
        properties1(m1,5)=stats(m1).Orientation;
        properties1(m1,6)=stats(m1).Centroid(1);
        properties1(m1,7)=stats(m1).Centroid(2);
        properties1(m1,8)=stats(m1).ConvexArea;   %number of pixels in image that specifies the convex hull
        properties1(m1,9)=stats(m1).Solidity;     %proportion of the pixels in the convex hull that are also in the region
        properties1(m1,10:13)=stats(m1,:).BoundingBox;
    end
end
end

% 
% set(gcf,'units','inches','position',[0,0,9,12])
% cd(strcat(HOME,'/output/centroids_ugly'))
% filename = [num2str(i),'_',fnm];
% set(gcf, 'PaperPositionMode', 'auto');
% print('-dtiff','-r250', filename)
% hold off;
% close;

%removes zero entry rows
prop = properties1(any(properties1,2),:);
[a,b]=size(prop);



%% plot nice image
im3 = BW_new;
[p,q]=size(im3);


% plot on raw images
cd(strcat(HOME,'/input/images'))
raw1 = imread(fnm);
%raw = imcrop(raw1,rect(crop_number,:)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
raw = raw1;                              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!

image1=raw(:,:,1);
image2=raw(:,:,2);
image3=raw(:,:,3);


cd(strcat(HOME,'/input/info'))
raw_black1=imread('black_raw.tif');
%raw_black = imcrop(raw_black1,rect(crop_number,:)); %%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
raw_black = raw_black1;                              %%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!


FLOE_LIBRARY1={};
floe=[];
floe2=[];
floe3=[];
if ~isempty(prop)
for t=1:a
im= imcrop(BW_new,prop(t,10:13));
outline = bwperim(im);
%figure, imshow(outline)

im_fill= imfill(outline,'holes');
%figure, imshow(im_fill)

im_fill2 = bwareaopen(im_fill, 50);
%figure, imshow(im_fill2)

floe = bwareafilt(im_fill2, 1, 'largest'); % keeps biggest floe in image
%figure, imshow(floe)

%only contours
floe2 =bwperim(floe);
floe3 = imdilate(floe2, true(2));

notzeros = find(floe==1);

new = zeros(size(BW_new));
Y1=prop(t,11);
Y2=prop(t,11)+prop(t,13);
X1=prop(t,10);
X2=prop(t,10)+prop(t,12);

    if Y2>p
       Y2=p;
    else
    end

    if X2>q
       X2=q;
    else
    end

new=zeros(p,q);
new(Y1:Y2, X1:X2) = floe;

new1=zeros(p,q);
new1(Y1:Y2, X1:X2) = floe3;

%plot on coast image
notzeros2 = find(new1==1);


image1(notzeros2)=0; %if =1 it will draw ice floe, if =0 it will delete it
image2(notzeros2)=0; %if =1 it will draw ice floe, if =0 it will delete it
image3(notzeros2)=0; %if =1 it will draw ice floe, if =0 it will delete it


%plot coastline
image1(filled_coastline==1)=0;
image2(filled_coastline==1)=0;
image3(filled_coastline==1)=0;

image_final=cat(3,image1,image2,image3); %concatenates the 3 channels to again form the rgb

FLOE_LIBRARY1{t,1}=floe;

%plot on coast image
raw_black(find(new==1))=1; %if =1 it will draw ice floe, if =0 it will delete it
          
end


else
image_final=raw;
end


FLOE_LIBRARY111{:,i1}=FLOE_LIBRARY1;
% figure,imshow(raw_black)
% figure,imshow(image_final)

%save image
cd(strcat(HOME,'/output/BW_nice'))
imwrite(raw_black,fullfile(fnm),'tiff', 'Compression','none')


%save image
cd(strcat(HOME,'/output/contours'))
imwrite(image_final,fullfile(fnm),'tiff', 'Compression','none')


end



% 
% for i=1:size(FLOE_LIBRARY111,2)
%     for ii=1:size(FLOE_LIBRARY111{1,i},1)
%     FLOE_LIBRARY{ii,i}=FLOE_LIBRARY111{:,i}{ii,1};
%     end
% end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('FINISHED PARFOR')
disp('BEGINING SEQUENCE')

%% begin to process slices of image
properties1=[];
properties={};

INDEX=0;
for zi=1:size(order,2)
i=order(zi);
INDEX=INDEX+1;

im=[];

%figure,imshow(BW4)
%save image
cd(strcat(HOME,'/output/BW_nice'))
fnm = im_files(i).name;
im = imread(fnm);
BW_binary = imread(fnm);


cd(strcat(HOME,'/output/contours'))
image_final= imread(fnm);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%BW4=binary;
BW_new=BW_binary;

BW_double=BW_new;

BW_new(find(filled==1))=0;
BW_double=double(BW_double);
red = BW_double; %red channel
green = BW_double; %blue channel
blue = BW_double; %greenchannel

%turns landmass grey
red(filled_coastline==1)=.75;
green(filled_coastline==1)=.75;
blue(filled_coastline==1)=.75;

BW_grey=cat(3,red,green,blue);
%figure,imshow(BW_grey)

%% obtain centroid
% properties matrix
figure, imshow(BW_grey), hold on
[B,~] = bwboundaries(BW_new,'noholes');
stats = regionprops(BW_new,'Area','Centroid','MajorAxisLength','MinorAxisLength',...
    'Orientation','Perimeter', 'BoundingBox', 'ConvexArea','Solidity');
[clust,numb] = bwlabel(BW_new, 8);
properties1 = zeros(numb, 13);
if numb~=0
%% set thresholds for centroid output:
threshold_min = 100;     %few= 5000
threshold_max = 90000;     %few=15000

%% for loop over all images and plot centroids
%stores: area, perimeter, major axis, minor axis, orientation, centroidx,
%centroidy in one matrix
for m = 1:numb
    boundary = B(m);
    area = stats(m).Area;
    major(m) = stats(m).MajorAxisLength;
    if area>threshold_min && area<threshold_max
        centroids = stats(m).Centroid;
        centroid(m,:) = centroids;
        plot(centroids(1),centroids(2),'ko', 'MarkerFaceColor','r','MarkerSize',6);   %'MarkerSize',8
        properties1(m,1)=stats(m).Area;
        properties1(m,2)=stats(m).Perimeter;
        properties1(m,3)=stats(m).MajorAxisLength;
        properties1(m,4)=stats(m).MinorAxisLength;
        properties1(m,5)=stats(m).Orientation;
        properties1(m,6)=stats(m).Centroid(1);
        properties1(m,7)=stats(m).Centroid(2);
        properties1(m,8)=stats(m).ConvexArea;   %number of pixels in image that specifies the convex hull
        properties1(m,9)=stats(m).Solidity;     %proportion of the pixels in the convex hull that are also in the region
        properties1(m,10:13)=stats(m,:).BoundingBox;
    end
end
else properties1(1,1:13)=zeros([1 13]);
end

%removes zero entry rows
properties2 = properties1(any(properties1,2),:);
properties{INDEX}=properties2;

prop=properties2;
[a,b]=size(prop);



%% axis data
[n,o,~]=size(BW_grey);
xlim([0 o])
ylim([0 n])

% automatically without 0
xticks(linspace(0,o,5))
xticklabels(LIST1(1:5))

yticks(linspace(0,n,5))
yticklabels(LIST2(1:5))

set(gca,'fontsize',14)
axis on


TITLE={};
if contains(fnm,'aqua')
TITLE= strcat(date_title(INDEX),':  Aqua');
elseif contains(fnm,'terra')
TITLE= strcat(date_title(INDEX),':  Terra');
end

hTitle  = title (TITLE);
hXLabel = xlabel(['\it{km} (', EXP2, ')']);
hYLabel = ylabel(['\it{km} (', EXP1, ')']);
set(gcf,'units','inches','position',[0,0,9,12])

%scale bar
rectangle('Position',[0,n-100,160,100],'FaceColor','black')
Scalebar_length = 80;   %20 km
quiver(35,n-70,Scalebar_length,0,'LineWidth',3, 'Color','white','ShowArrowHead','off','AutoScale','off')
text(25,n-30,'20 km','Color','white','FontSize',12)

%Georgy region
% hTitle  = title (TITLE);
% hXLabel = xlabel(['\it{km} (', EXP2, ')']);
% hYLabel = ylabel(['\it{km} (', EXP1, ')']);
% set(gcf,'units','inches','position',[0,0,12,8])
%
% %scalebar
% rectangle('Position',[10,n-75,100,75],'FaceColor','black')
% Scalebar_length = 80;   %40 km
% quiver(20,n-50,Scalebar_length,0,'LineWidth',3, 'Color','white','ShowArrowHead','off','AutoScale','off')
% text(20,n-20,'20 km','Color','white','FontSize',12)


%print images with centroids
cd(strcat(HOME,'/output/centroids_nice'))
print('-dtiff','-r250',fnm)
hold off;
close;


%% plot centroids on raw images
figure, imshow(image_final), hold on
[B,L] = bwboundaries(BW_new,'noholes');
stats = regionprops(BW_new,'Area','Centroid','MajorAxisLength','MinorAxisLength',...
    'Orientation','Perimeter', 'BoundingBox', 'ConvexArea','Solidity');
[clust,numb] = bwlabel(BW_new, 8);

%% set thresholds for centroid output:
threshold_min = 100;     %few= 5000
threshold_max = 90000;     %few=15000

%% for loop over all images and plot centroids
%stores: area, perimeter, major axis, minor axis, orientation, centroidx,
%centroidy in one matrix
if numb~=0
for m = 1:numb
    boundary = B(m);
    area = stats(m).Area;
    major(m) = stats(m).MajorAxisLength;
    if area>threshold_min && area<threshold_max
        centroids = stats(m).Centroid;
        centroid(m,:) = centroids;
        plot(centroids(1),centroids(2),'o', 'MarkerSize',4,...
        'MarkerEdgeColor','red','MarkerFaceColor','red');
    end
end
end
cd(strcat(HOME,'/output/centroids'))
filename = [num2str(INDEX),'_',fnm];
print('-dtiff','-r250', filename)
hold off;
close all;
end

cd(strcat(HOME,'/output/tracked'))
save('prop','properties');




clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
cd(strcat(HOME,'/output/tracked'))
properties=importdata('prop.mat');
order=importdata('order.mat');

cd(strcat(HOME,'/output/BW_nice'))
im_files = dir('*.tif');

INDEX=0;
parfor (i1=1:size(order,2),M_WORKERS) %1:size(order,2)
i=order(i1)
INDEX=INDEX+1;

BW_new=[];

%figure,imshow(BW4)
%save image
cd(strcat(HOME,'/output/BW_nice'))
fnm = im_files(i).name;
BW_new = imread(fnm);

%removes zero entry rows
prop = properties{1,i};
[a,b]=size(prop);

%% plot nice image
[p,q]=size(BW_new);



FLOE_LIBRARY1={};
floe=[];
if ~isempty(prop)
for t=1:a
im= imcrop(BW_new,prop(t,10:13));
outline = bwperim(im);
%figure, imshow(outline)

im_fill= imfill(outline,'holes');
%figure, imshow(im_fill)

im_fill2 = bwareaopen(im_fill, 50);
%figure, imshow(im_fill2)

floe = bwareafilt(im_fill2, 1, 'largest'); % keeps biggest floe in image
%figure, imshow(floe)

%only contours
floe2 =bwperim(floe);
floe3 = imdilate(floe2, true(2));

notzeros = find(floe==1);

new = zeros(size(BW_new));
Y1=prop(t,11);
Y2=prop(t,11)+prop(t,13);
X1=prop(t,10);
X2=prop(t,10)+prop(t,12);

    if Y2>p
       Y2=p;
    else
    end

    if X2>q
       X2=q;
    else
    end

new=zeros(p,q);
new(Y1:Y2, X1:X2) = floe;

FLOE_LIBRARY1{t,1}=floe;

end

end


FLOE_LIBRARY111{:,i1}=FLOE_LIBRARY1;
% figure,imshow(raw_black)
end


for i=1:size(FLOE_LIBRARY111,2)
    for ii=1:size(FLOE_LIBRARY111{1,i},1)
    FLOE_LIBRARY{ii,i}=FLOE_LIBRARY111{:,i}{ii,1};
    end
end

cd(strcat(HOME,'/output/tracked'))
save('FLOE_LIBRARY.mat', 'FLOE_LIBRARY', '-v7.3')

disp('FINISHED IMAGE PROCESSING')
disp('BEGINING TRACKER')








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRACKER
%Identifies and tracks ice floes in consecutive days. By matching major and
%minor axis, and constraining the traveled distance, each ice floe is id.
%Also gets rid of repeated ice floes that were accounted more than once by
%matching ice floes to those with the minimum distance traveled

%% identify ice floes by matching major and minor axes in consecutive images
% properties= [area, perimeter, major axis, minor axis, orientation,
% centroidx,centroidy]
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all
tic

cd(strcat(HOME,'/output/tracked'))
order = importdata('order.mat');  
delta_t = importdata('delta_t.mat');  
prop = importdata('prop.mat');  

[a,b]= size(prop);   
final_data={};


matrix1=[];
matrix2=[];
data1=[];
data2=[];
percent=[];

z_tracker=[];

int1=round(size(prop,2)/M_WORKERS);
if int1<=1
    int1=2;
end
list1=1:int1:size(prop,2);
list2=[];
for i=1:size(list1,2)
list2(i,:)=list1(i):list1(i)+int1;
end
list2(:,end)=[];

M=size(list1,2);
N=size(list2,2);

list2(list2>=size(prop,2))=NaN;
parfor z=1:M
for z_idx1=1:N
z_idx=list2(z,z_idx1);
if ~isnan(z_idx)    
matrix1=prop{z_idx};
matrix2=prop{z_idx+1};
delta_time=delta_t(z_idx);

idx=1;
old=rand(1,15);
match_total=[];
data1=[];

while ~isequal(old,matrix1)
idx=1;
match=[];
match1=[];
match2=[];
for r = 1:size(matrix1,1)   
    data=[];   
    percent=[];
    index=1;
    pt1=matrix1(r,6:7);
    dist1=pdist2(pt1,matrix2(:,6:7));
    close_pts=find(dist1<250);
    
if ~isempty(close_pts)
    for s = close_pts
       point1=[matrix1(r,6),matrix1(r,7)];
       point2=[matrix2(s,6),matrix2(s,7)];
       
    if ( delta_time < 30 && abs(pdist2(point1,point2))<15 ) ||...
       ( delta_time >= 30 && delta_time <= 100 &&  abs(pdist2(point1,point2))<30 ) || ...                                        %distance
       ( delta_time >= 1300 && abs(pdist2(point1,point2))<120 )
       if matrix1(r,1)>1200
          if abs((matrix1(r,1)-matrix2(s,1))/(.5*(matrix1(r,1)+matrix2(s,1))))<0.28 ...     %area    .05
          &&  abs((matrix1(r,3)-matrix2(s,3))/(.5*(matrix1(r,3)+matrix2(s,3))))<0.10 ... %major axis .09 
          &&  abs((matrix1(r,4)-matrix2(s,4))/(.5*(matrix1(r,4)+matrix2(s,4))))<0.12...  %minor axis .11 
          &&  abs((matrix1(r,8)-matrix2(s,8))/(.5*(matrix1(r,8)+matrix2(s,8))))<0.14     %convex area.12 
         
          cd(strcat(HOME,'/input/info'))
          [area_under,correlation]=match_corr(r,s,z_idx); 
          
          if ~isnan(area_under) && area_under<.236 && correlation>.68
          data(index,:) = matrix2(s,:);
          percent(index,1) = abs(matrix1(r,1)-matrix2(s,1))/(.5*(matrix1(r,1)+matrix2(s,2)));  %area
          percent(index,2) = abs(matrix1(r,3)-matrix2(s,3))/(.5*(matrix1(r,3)+matrix2(s,3)));  %maj axis
          percent(index,3) = abs(matrix1(r,3)-matrix2(s,3))/(.5*(matrix1(r,3)+matrix2(s,3)));  %min axis
          percent(index,4) = area_under;                                                       %matching area
          percent(index,5) = 1-correlation;                                                    %corr coeff
          index=index+1; 
          end
          end
       
       elseif matrix1(r,1)<=1200 
          if abs((matrix1(r,1)-matrix2(s,1))/(.5*(matrix1(r,1)+matrix2(s,1))))<0.18 ...     %area
          &&  abs((matrix1(r,3)-matrix2(s,3))/(.5*(matrix1(r,3)+matrix2(s,3))))<0.07 ... %major axis
          &&  abs((matrix1(r,4)-matrix2(s,4))/(.5*(matrix1(r,4)+matrix2(s,4))))<0.08...  %minor axis
          &&  abs((matrix1(r,8)-matrix2(s,8))/(.5*(matrix1(r,8)+matrix2(s,8))))<0.09     %convex area
                         
          cd(strcat(HOME,'/input/info'))
          [area_under,correlation]=match_corr(r,s,z_idx); 
          
          if ~isnan(area_under) && area_under<.18 && correlation>.68
          data(index,:) = matrix2(s,:);
          percent(index,1) = abs(matrix1(r,1)-matrix2(s,1))/(.5*(matrix1(r,1)+matrix2(s,2)));  %area
          percent(index,2) = abs(matrix1(r,3)-matrix2(s,3))/(.5*(matrix1(r,3)+matrix2(s,3)));  %maj axis
          percent(index,3) = abs(matrix1(r,3)-matrix2(s,3))/(.5*(matrix1(r,3)+matrix2(s,3)));  %min axis
          percent(index,4) = area_under;                                                       %matching area
          percent(index,5) = 1-correlation;                                                    %corr coeff
          index=index+1;    
          end
          end
          
          
       end
       
    end
    end
 
    position=[];
    value=[];
    place=[];
    % now select option that is MOST minimum everything
    [c,d]=size(percent);
    for i=1:d
    [value, place]= min(percent(:,i));
    position(i) = place;
    end
    value1=mode(position);
    if ~isempty(position) 
    match(idx,:,1)= [matrix1(r,:),percent(value1,4:5)];
    match(idx,:,2)= [data(value1,:),0,0];
    match1(idx,:,1)= match(idx,1:13,1);
    match2(idx,:,2)= match(idx,1:13,2);
        
    idx=idx+1;
    end       
end 
end
 
if isempty(match)
   old=matrix1; 
   break
else 


row=[];
y=[];
value=[];
place=[];
%are there a lot repeated?
for j=1:size(match(:,:,2),1)
% is row j repeated?
[row] = ismember(match(:,:,2), match(j,:,2), 'rows');  %find repeated rows
% locb is a logival vector.  Find the actual row numbers.

y=find(row==1);
A=[];
B=[];
percent=[];

%pick smallest and pair up, otherwise set to 0 for all others
for f=1:size(y,1)
    A=match(y(f),:,1);
    B=match(y(f),:,2);
      percent(f,1) = abs(A(1,1)-B(1,1))/(.5*(A(1,1)+B(1,1)));   %area
      percent(f,2) = abs(A(1,3)-B(1,3))/(.5*(A(1,3)+B(1,3)));   %major axis
      percent(f,3) = abs((A(1,4)-B(1,4))/(.5*(A(1,4)+B(1,4)))); %minor axis
      percent(f,4) = abs(A(1,14));                              %area under
      percent(f,5) = abs(A(1,15));                              %corr
end 

position=[];
% now select option that is MOST minimum everything
[c,d]=size(percent);
value=[];
place=[];
for i=1:d
[value, place]= min(percent(:,i));
position(i) = place;
end
value1=mode(position);


%now that value1 is the best match set all others to 0 
a=match(y(value1),:,1);

for g=1:size(y,1)
if all(match(y(g),2:4,1)~=a(:,2:4))
   match(y(g),:,:)=0;
end       
end

end

new_matrix1=[];
new_matrix2=[];
% run again and match left overs
% % % % % % % % % % % % % % % % % % % % save new matrix information
% % % % % % % % % % % % % % % % % % % for i=1:size(matrix1,1) 
% % % % % % % % % % % % % % % % % % %     if  all(~ismember(match1(:,:,1),matrix1(i,:),'rows'))
% % % % % % % % % % % % % % % % % % %         new_matrix1(i,:)  = matrix1(i,:);         
% % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % for i=1:size(matrix2,1) 
% % % % % % % % % % % % % % % % % % %     if all(~ismember(match2(:,1,2),matrix2(i,1),'rows')) 
% % % % % % % % % % % % % % % % % % %         new_matrix2(i,:)  = matrix2(i,:);        
% % % % % % % % % % % % % % % % % % %     end
% % % % % % % % % % % % % % % % % % % end


old=matrix1;

%get rid of zero rows
new_matrix1x=new_matrix1;
new_matrix2x=new_matrix2;

new_matrix1x(~any(new_matrix1,2),:) = [];
new_matrix2x(~any(new_matrix2x,2),:) = [];

matrix1=[];
matrix2=[];
matrix1=new_matrix1x;
matrix2=new_matrix2x;

end
if ~isequal(old,matrix1)
[aa,bb]=size(match);
[cc,dd]=size(match_total);
match_total(cc+1:cc+aa,:,:)=match;
match_total;
end
   
end

if ~isempty(match_total)
match_final=[];

%get rid of zero rows
match_totalx = match_total(:,:,1);
match_totalx2= match_total(:,:,1);
match_totaly = match_total(:,:,2);
match_totaly2= match_total(:,:,2);

match_totalx2(~any(match_totalx,2),:) = [];
match_totaly2(~any(match_totaly,2),:) = [];

match_final(:,:,1)=match_totalx2;
match_final(:,:,2)=match_totaly2;


day=[];
d=[];
for p= 1:size(match_final,1)
    day(p)=z_idx;
end    

for mm = 1:size(match_final,1)
centroid1 = [match_final(mm,6,1),match_final(mm,7,1)];
centroid2 = [match_final(mm,6,2),match_final(mm,7,2)];
    d(mm) = pdist([centroid1;centroid2],'euclidean');
end

data1(:,1)= day;
data1(:,2)= match_final(:,1,1);
data1(:,3)= match_final(:,1,2);
data1(:,4)= match_final(:,2,1);
data1(:,5)= match_final(:,2,2);
data1(:,6)= match_final(:,3,1);
data1(:,7)= match_final(:,3,2);
data1(:,8)= match_final(:,4,1);
data1(:,9)= match_final(:,4,2);
data1(:,10)=match_final(:,6,1);
data1(:,11)=match_final(:,6,2);
data1(:,12)=match_final(:,7,1);
data1(:,13)=match_final(:,7,2);
data1(:,14)=d;

end
%% organizes data by decreasing area
if isempty(data1)
   data2= zeros([1,14]);
   old_data1{z,z_idx1}=data2;
else    
   data2 = sortrows(data1,-2);
   old_data1{z,z_idx1}=data2;
end
end
end
end


IDX=0;
for i=1:size(old_data1,1) 
for jj=1:size(old_data1,2)    
IDX=IDX+1;
old_data{1,IDX}=old_data1{i,jj};
end
end

toc
cd(strcat(HOME,'/output/tracked'))
save(['old_data'],'old_data');
disp('TRACKER 1')


%% LONG TRACKER
%Creates cube of information as the following: Heigth(individual ice floe)- 
%contains different ice floes organized by decreasing area. Length
%(properties)- stores properties of each ice floe in the following way: 
%area, perimeter, major axis, minor axis, x-coord and y-coord centroid. 
%Width(time)-the first matrix contains data from Day1, the second matrix
%has data from Day2,etc. Each ice floe is stacked directly behind
%corresponding ice floe from previous day.

clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all
cd(strcat(HOME,'/output/tracked'))
data = importdata('old_data.mat');  

%% get Day1 information 
[a,b]=size(data{1,1});

A1= [data{1,1}(:,2),data{1,1}(:,4), data{1,1}(:,6), data{1,1}(:,8),...
     data{1,1}(:,10), data{1,1}(:,12), zeros([a 1]), zeros([a 1]), zeros([a 1])];


[aa,bb]=size(A1);

rows=ROWS-a;

A1= [A1; zeros([rows bb])];

%% set up 3D matrix and correct shape for data cell
A2= zeros([ROWS bb]);
cd(strcat(HOME,'/output/masked'))
%cd('/Users/rosalinda/Documents/UCR/Research/ice_tracker/branch1/FINAL/input/images')
files = dir('*.tif');

days=size(files,1); %change depending on days
tracker(:,:,days)= A2;

new_data={A2, data{1,:}};

[c,d]= size(new_data);  

counter=0;


%% loops over all of new_data to produce info cube where individual ice floes are stacked behind each other
for k=1:d-1    
matrix1=tracker(:,:,k);
matrix2=new_data{k+1};

[e,f]= size(matrix1);
[g,h]= size(matrix2);


if all(all(matrix2==0))
B0= zeros([size(A1)]);
tracker_previous = tracker(:,:,k);
tracker_current= zeros(e,f);
l=0;
else

B0=[];
B1=[];


B0= [matrix2(:,2),matrix2(:,4),matrix2(:,6),matrix2(:,8),matrix2(:,10),matrix2(:,12)]; %area2,perim2,maj2,min2,x2,y2
B0= [B0 zeros(size(B0,1),3)];

B1= [matrix2(:,3),matrix2(:,5),matrix2(:,7),matrix2(:,9),matrix2(:,11),matrix2(:,13)]; %area2,perim2,maj2,min2,x2,y2
B1= [B1 zeros(size(B1,1),3)];


area0 = [];
perim0 = [];
maj0 = [];
min0 = [];
x_coord0 = [];
y_coord0 = [];

area1 = [];
perim1 = [];
maj1 = [];
min1 = [];
x_coord1 = [];
y_coord1 = [];

for i=1:e
for j=1:g
    if matrix1(i,5)==matrix2(j,10) && matrix1(i,6)==matrix2(j,12)
       tracker(i,1:6,k) = B0(j,1:6);
       tracker(i,1:6,k+1) = B1(j,1:6);
    end
end
end

%for ice floes that were not identified on previous days they are listed at
%the bottom 
tracker_previous=[];
tracker_current=[];
tracker_previous = tracker(:,:,k);
tracker_current= tracker(:,:,k+1);

member= ~ismember(B0,tracker_previous,'rows');  
non_repeated0= B0(member==1,:);
non_repeated1= B1(member==1,:);

[l,m]=size(non_repeated1);

tracker_previous(counter+1:counter+l,:)=non_repeated0(1:l,:);   
tracker_current(counter+1:counter+l,:)=non_repeated1(1:l,:);   
end

tracker(:,:,k)=tracker_previous;
tracker(:,:,k+1)=tracker_current;
counter=counter+l;
end


cd(strcat(HOME,'/output/tracked'))
save('tracker','tracker');

disp('FINISHED TRACKER')
disp('BEGINING ONE-DAY SKIP')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% skip one day (My routine)
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

cd(strcat(HOME,'/output/tracked'))
final_tracker = importdata('tracker.mat');

%stores: area, perimeter, major axis, minor axis, centroidx, centroidy,
%0,0,0


prop=importdata('prop.mat');
%stores: area, perimeter, major axis, minor axis, orientation, centroidx,
%centroidy, solidity, convex area, bounding boxin one matrix

delta_t=importdata('delta_t.mat');

% x= importdata('x.mat');
% y= importdata('y.mat');


[a,b,c ]=size(final_tracker);
[aa,bb]=size(prop);
day1=[];
day2=[];
prop_3=[];
prop1=[];


%% select ice floes to replace
for j = 5:bb %day
%final_tracker = importdata('tracker_skip.mat');
prop_1=[];
prop_11=[];
prop_111=[];
PROP1=[];
all_day5=[];


day1=final_tracker(:,:,j-4);
day2=final_tracker(:,:,j-3);
day3=final_tracker(:,:,j-2);
day4=final_tracker(:,:,j-1);
day5=final_tracker(:,:,j);

% [e,~]=find(day5~=0);
[e11,~]=find(day1~=0);
[e22,~]=find(day2~=0);
[e33,~]=find(day3~=0);
[e44,~]=find(day4~=0);

e1=unique(e11);
e2=unique(e22);
e3=unique(e33);
e4=unique(e44);


prop1=prop{:,j};
prop_1=prop1;

%deletes ice floes already matched from props and creates new matrix to
%match leftover ice floes with skip routine
for z=1:size(day3,1)
    if day5(z,1)~=0 
        [w,ww]=ismember(day5(z,5),prop1(:,6));
        prop_1(ww(1),:)=zeros([1,13]);
    end
end    

%get rid of zero rows
prop_11=prop_1;
prop_11(~any(prop_1,2),:) = [];

list=[1,2,3,4,6,7];
prop_111=prop_11(:,list);
f=size(prop_111,1);
B=zeros([f,3]);
PROP1=[prop_111 B];

if isempty(PROP1)
   all_day5=day5;
   [m1,~]=find(all_day5~=0);
   m=unique(m1);
else   
    
%make new matrix with matched and non-matched floes to make good skip matches
all_day5=day5;

all_day5(a-size(PROP1,1)+1:a,:)=PROP1;
[m1,~]=find(all_day5~=0);
m=unique(m1);
end

DAY1=[];
DAY2=[];
DAY11=[];
DAY22=[];
DAY111=[];
DAY222=[];
index=1;

rows=[];
rowss=[];
option=[];

delta_t4=delta_t(j-1);
    if delta_t4<30
    d4=15;     %dist
    elseif delta_t4>30 && delta_t4<100
    d4=20;     %dist
    elseif delta_t4>1300
    d4=120;    %dist
    end    

delta_t3=delta_t(j-1)+delta_t(j-2);
    if delta_t3<30
    d3=15;     %dist    
    elseif delta_t3>30 && delta_t3<100
    d3=20;     %dist
    elseif delta_t3>=1300 && delta_t3<=2500 
    d3=120;    %dist
    elseif delta_t3>2500 
    d3=240;    %dist
    end
    
delta_t2=delta_t(j-1)+delta_t(j-2)+delta_t(j-3);
    if delta_t2<30
    d2=15;     %dist
    elseif delta_t2>30 && delta_t2<100
    d2=30;     %dist
    elseif delta_t2>1300 && delta_t2<2500 
    d2=120;    %dist   
    elseif delta_t2>2500 
    d2=240;    %dist
    end
       
delta_t1=delta_t(j-1)+delta_t(j-2)+delta_t(j-3)+delta_t(j-4);
    if delta_t1>30 && delta_t1<100
    d1=30;     %dist
    elseif delta_t1>1300 && delta_t1<2500
    d1=120;    %dist
    elseif delta_t1>2500 
    d1=240;    %dist
    end
    
for k = m'    %row on day5
ee=[];
ee1_dist=pdist2(all_day5(k,5:6),day1(e1,5:6));
closest_e1=e1(find(ee1_dist<700));

ee2_dist=pdist2(all_day5(k,5:6),day2(e2,5:6));
closest_e2=e2(find(ee2_dist<700));

ee3_dist=pdist2(all_day5(k,5:6),day3(e3,5:6));
closest_e3=e3(find(ee3_dist<700));

ee4_dist=pdist2(all_day5(k,5:6),day4(e4,5:6));
closest_e4=e4(find(ee4_dist<700));

ee=[closest_e1;closest_e2;closest_e3;closest_e4];

for i = ee'    %row on day3 
    if day5(k,5)~=0
    if ( (abs(all_day5(k,5)- day4(i,5))<240 && abs(all_day5(k,6)- day4(i,6))<240)...
    || (abs(all_day5(k,5)- day3(i,5))<240 && abs(all_day5(k,6)- day3(i,6))<240)...
    || (abs(all_day5(k,5)- day2(i,5))<200 && abs(all_day5(k,6)- day2(i,6))<200)...
    || (abs(all_day5(k,5)- day1(i,5))<200 && abs(all_day5(k,6)- day1(i,6))<200) )
    DAY_5= all_day5(k,1:9);
    point5=[all_day5(k,5),all_day5(k,6)];
       
      if day4(i,5)~=0
         DAY_4=day4(i,1:9);
         point4=[day4(i,5),day4(i,6)];
         prop_4=prop{:,j-1};
      else
         point4=point5;
         DAY_4=DAY_5;
      end
      if day3(i,5)~=0
         DAY_3=day3(i,1:9);
         point3=[day3(i,5),day3(i,6)];
         prop_3=prop{:,j-2};
      else 
         point3=point5;
         DAY_3=DAY_5;
      end
      if day2(i,5)~=0
         DAY_2=day2(i,1:9);
         point2=[day2(i,5),day2(i,6)];
         prop_2=prop{:,j-3};
      else 
         point2=point5;
         DAY_2=DAY_5;
      end
      if day1(i,5)~=0
         DAY_1=day1(i,1:9);
         point1=[day1(i,5),day1(i,6)];
         property_1=prop{:,j-4};
      else
         point1=point5;
         DAY_1=DAY_5;
      end
     
 if all_day5(k,1)>1200                                                            
  if day4(k,1)==0 && all_day5(i,1)==0 ...  %no replacing existing ice floes  
  && abs(pdist2(point4,point5))<d4 ...                    %distance (1 day)
  && abs(pdist2(point3,point5))<d3 ...                    %distance (1 day)
  && abs(pdist2(point2,point5))<d2 ...                   %distance (2 days)
  && abs(pdist2(point1,point5))<d1                       %distance (2 days)

   if abs((DAY_4(1,1)-all_day5(k,1))/(0.5*(DAY_4(1,1)+all_day5(k,1))))<0.28 ...  %DAY4: area  
   && abs((DAY_4(1,3)-all_day5(k,3))/(0.5*(DAY_4(1,3)+all_day5(k,3))))<0.10 ...  %maj_axis 
   && abs((DAY_4(1,4)-all_day5(k,4))/(0.5*(DAY_4(1,4)+all_day5(k,4))))<0.12 ...  %min_axis 
   && abs((DAY_3(1,1)-all_day5(k,1))/(0.5*(DAY_3(1,1)+all_day5(k,1))))<0.28 ...  %DAY3: area
   && abs((DAY_3(1,3)-all_day5(k,3))/(0.5*(DAY_3(1,3)+all_day5(k,3))))<0.10 ...  %maj_axis  
   && abs((DAY_3(1,4)-all_day5(k,4))/(0.5*(DAY_3(1,4)+all_day5(k,4))))<0.12 ...  %min_axis 
   && abs((DAY_2(1,1)-all_day5(k,1))/(0.5*(DAY_2(1,1)+all_day5(k,1))))<0.28 ...  %DAY2: area
   && abs((DAY_2(1,3)-all_day5(k,3))/(0.5*(DAY_2(1,3)+all_day5(k,3))))<0.10 ...  %maj_axis 
   && abs((DAY_2(1,4)-all_day5(k,4))/(0.5*(DAY_2(1,4)+all_day5(k,4))))<0.12 ...  %min_axis 
   && abs((DAY_1(1,1)-all_day5(k,1))/(0.5*(DAY_1(1,1)+all_day5(k,1))))<0.28 ...  %DAY1: area
   && abs((DAY_1(1,3)-all_day5(k,3))/(0.5*(DAY_1(1,3)+all_day5(k,3))))<0.10 ...  %maj_axis 
   && abs((DAY_1(1,4)-all_day5(k,4))/(0.5*(DAY_1(1,4)+all_day5(k,4))))<0.12      %min_axis  
          
     if day4(i,5)~=0 && ...                                          %first
     abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-prop_4(find(prop_4(:,3)==day4(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+prop_4(find(prop_4(:,3)==day4(i,3)),8))))<0.16 %convex 
 
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(prop_4(:,3)==day4(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),j-1);
     if area_under<.236 && correlation>.68
     DAY1(index,1:9) = day4(i,1:9);
     DAY1(index,10) = 1;                                            %option
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
                
     elseif day3(i,5)~=0 ...                                        %second  
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-prop_3(find(prop_3(:,3)==day3(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+prop_3(find(prop_3(:,3)==day3(i,3)),8))))<0.16 %convex  
    
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(prop_3(:,3)==day3(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),[j-2,j]); 
     if area_under<.236 && correlation>.68                                
     DAY1(index,1:9) = day3(i,1:9);
     DAY1(index,10) = 2;                                            %option
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
                
     elseif day3(i,5)==0 && day2(i,5)~=0 ...                         %third
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-prop_2(find(prop_2(:,3)==day2(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+prop_2(find(prop_2(:,3)==day2(i,3)),8))))<0.16 %convex 18%
 
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(prop_2(:,3)==day2(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),[j-3,j]);  
     if area_under<.236 && correlation>.68                                
     DAY1(index,1:9) = day2(i,1:9);
     DAY1(index,10) = 3;                                            %option
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
                
     elseif day3(i,5)==0 && day2(i,5)==0 && day1(i,5)~=0 ...        %fourth 
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-property_1(find(property_1(:,3)==day1(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+property_1(find(property_1(:,3)==day1(i,3)),8))))<0.16 %convex 18% 

     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(property_1(:,3)==day1(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),[j-4,j]);  
     if area_under<.236 && correlation>.68                                                        
     DAY1(index,1:9) = day1(i,1:9);
     DAY1(index,10) = 4;                                            %option               
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
     end
    end
  end
  
 elseif all_day5(k,1)<1200
  if day4(k,1)==0 && all_day5(i,1)==0 ...  %no replacing existing ice floes  
  && abs(pdist2(point4,point5))<d4 ...     %distance (1 day)
  && abs(pdist2(point3,point5))<d3 ...     %distance (1 day)
  && abs(pdist2(point2,point5))<d2 ...     %distance (2 days)
  && abs(pdist2(point1,point5))<d1         %distance (2 days)


   if abs((DAY_4(1,1)-all_day5(k,1))/(0.5*(DAY_4(1,1)+all_day5(k,1))))<0.18 ...  %DAY4: area  
   && abs((DAY_4(1,3)-all_day5(k,3))/(0.5*(DAY_4(1,3)+all_day5(k,3))))<0.07 ...  %maj_axis  
   && abs((DAY_4(1,4)-all_day5(k,4))/(0.5*(DAY_4(1,4)+all_day5(k,4))))<0.08 ...  %min_axis 
   && abs((DAY_3(1,1)-all_day5(k,1))/(0.5*(DAY_3(1,1)+all_day5(k,1))))<0.18 ...  %DAY3: area  
   && abs((DAY_3(1,3)-all_day5(k,3))/(0.5*(DAY_3(1,3)+all_day5(k,3))))<0.07...   %maj_axis 
   && abs((DAY_3(1,4)-all_day5(k,4))/(0.5*(DAY_3(1,4)+all_day5(k,4))))<0.08 ...  %min_axis 
   && abs((DAY_2(1,1)-all_day5(k,1))/(0.5*(DAY_2(1,1)+all_day5(k,1))))<0.18 ...  %DAY2: area 
   && abs((DAY_2(1,3)-all_day5(k,3))/(0.5*(DAY_2(1,3)+all_day5(k,3))))<0.07 ...  %maj_axis  
   && abs((DAY_2(1,4)-all_day5(k,4))/(0.5*(DAY_2(1,4)+all_day5(k,4))))<0.08 ...  %min_axis   
   && abs((DAY_1(1,1)-all_day5(k,1))/(0.5*(DAY_1(1,1)+all_day5(k,1))))<0.18 ...  %DAY1: area 
   && abs((DAY_1(1,3)-all_day5(k,3))/(0.5*(DAY_1(1,3)+all_day5(k,3))))<0.07 ...  %maj_axis
   && abs((DAY_1(1,4)-all_day5(k,4))/(0.5*(DAY_1(1,4)+all_day5(k,4))))<0.08      %min_axis 

     if day4(i,5)~=0 ...                                             %first  
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-prop_4(find(prop_4(:,3)==day4(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+prop_4(find(prop_4(:,3)==day4(i,3)),8))))<0.09 %convex  
      
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(prop_4(:,3)==day4(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),j-1);
     if area_under<.18 && correlation>.68
     DAY1(index,1:9) = day4(i,1:9);
     DAY1(index,10) = 1;                                            %option
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
                
     elseif day3(i,5)~=0 ...                                        %second  
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-prop_3(find(prop_3(:,3)==day3(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+prop_3(find(prop_3(:,3)==day3(i,3)),8))))<0.09 %convex 
     
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(prop_3(:,3)==day3(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),[j-2,j]); 
     if area_under<.18 && correlation>.68
     DAY1(index,1:9) = day3(i,1:9);
     DAY1(index,10) = 2;                                            %option
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
                
     elseif day3(i,5)==0 && day2(i,5)~=0 ...                         %third
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-prop_2(find(prop_2(:,3)==day2(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+prop_2(find(prop_2(:,3)==day2(i,3)),8))))<0.09 %convex 
      
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(prop_2(:,3)==day2(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),[j-3,j]);
     if area_under<.18 && correlation>.68                               
     DAY1(index,1:9) = day2(i,1:9);
     DAY1(index,10) = 3;                                            %option
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
                
     elseif day3(i,5)==0 && day2(i,5)==0 && day1(i,5)~=0 ...        %fourth
     && abs((prop1(find(prop1(:,3)==all_day5(k,3)),8)-property_1(find(property_1(:,3)==day1(i,3)),8))/...
     (0.5*(prop1(find(prop1(:,3)==all_day5(k,3)),8)+property_1(find(property_1(:,3)==day1(i,3)),8))))<0.09 %convex 
  
     cd(strcat(HOME,'/input/info'))
     [area_under,correlation]=match_corr(find(property_1(:,3)==day1(i,3)),...
                              find(prop1(:,3)==all_day5(k,3)),[j-4,j]);
     if area_under<.18 && correlation>.68
     DAY1(index,1:9) = day1(i,1:9);
     DAY1(index,10) = 4;                                           %option               
     DAY2(index,1:9) = all_day5(k,1:9); 
     DAY2(index,9) = area_under;
     index=index+1;
     end
     continue
     end
   end
   end
 end
    
    end
    end
 
end
end

if isempty(DAY1)
    continue
%find repeated rows    
else    
[val, ~, ic] = unique(DAY2(:,5));
min_dist = accumarray(ic(:), DAY2(:,9), [], @min);
min_val = [val(:), min_dist(:)]; 

[l,~]= size(min_val);
idx=1;

%select most minimum
for h=1:l
    [not_repeated,z]=find(DAY2(:,9)==min_val(h,2));
    rows(idx)=not_repeated(1);
    idx=idx+1;
end

rows=rows';

DAY11=DAY1(rows,:);
DAY22=DAY2(rows,:);

[val, ~, ic] = unique(DAY11(:,5));
min_dist = accumarray(ic(:), DAY22(:,9), [], @min);
min_val = [val(:), min_dist(:)]; 
[l,m]= size(min_val);
idx=1;


for h=1:l
    [not_repeated,z]=find(DAY22(:,9)==min_val(h,2));
    rowss(idx)=not_repeated(1);
    idx=idx+1;
end

rowss=rowss';

DAY111=DAY11(rowss,:);
DAY222=DAY22(rowss,:);

old_row=[];
new_row=[];
old=[];
new=[];
delete=[];

for h=1:size(DAY222,1)
    [old_row,~] = find(final_tracker(:,5,j)==DAY222(h,5));
    [new_row,~] = find(final_tracker(:,5,j-DAY111(h,10))==DAY111(h,5));  
    if ~isempty(old_row)%ismember(DAY222(h,5),final_tracker(:,5,j))
       final_tracker(new_row,:,j:c) = final_tracker(old_row,:,j:c);
       delete=final_tracker(old_row,:,j:c); 
       final_tracker(old_row,:,j:c)=zeros(size(delete)); 
       fprintf('replaced row %d \ton day %d\twith row\t%d\tfrom %d time steps prior\n',new_row,j,old_row,DAY111(h,10));
    else
       final_tracker(new_row,:,j) = DAY222(h,:);
       fprintf('replaced row %d \ton day %d\tfrom unmmatched ice floe\n',new_row,j);
    end
    
cd(strcat(HOME,'/output/tracked'))
save('tracker_skip','final_tracker');
end
end

cd(strcat(HOME,'/output/tracked'))
z_long(j,1)=j;
save('z_long','z_long')
end
cd(strcat(HOME,'/output/tracked'))
save('tracker_skip','final_tracker');

cd(strcat(HOME,'/output/tracked'))
tracker2 = importdata('tracker_skip.mat');  



%% calculate the distance traveled by ice floe from point 1 to point 2.
% Column 9 stores this value and shows after the distance is travelled (so at point 2)
final_tracker= tracker2;
[n,o,p]=size(final_tracker);
mat1=[];
mat2=[];
for k=1:p-1
mat1=final_tracker(:,:,k+1);
mat2=final_tracker(:,:,k);
[q,r]= size(mat1);

cent1=[];
cent2=[];
    for i=1:q
        if mat1(i,6)~=0 && mat2(i,6)~=0
            cent1=transpose([mat1(i,5);mat1(i,6)]);
            cent2=transpose([mat2(i,5);mat2(i,6)]);

            mat1(i,9) = pdist([cent1;cent2],'euclidean');    
            final_tracker(:,:,k+1) = mat1;  
        else
        end
    end

end
%% save tracking properites in a matrix
%area, perim, maj_axis, min_axis, x_cent,y_cent,lat,long,dist
cd(strcat(HOME,'/output/tracked'))
save(['tracker_distance'],'final_tracker');


%% save tracking properites in meters (except for centroid coordinates)
%area,perim,maj_axis,min_axis,x_cent,y_cent,lat,long,dist


[a,b,c]=size(final_tracker);
for k=1:c
    for i=1:a
    final_tracker(i,1,k)=final_tracker(i,1,k)*.250^2;
    final_tracker(i,2:4,k)=final_tracker(i,2:4,k)*.250;
    final_tracker(i,9,k)=final_tracker(i,9,k)*.250;
    end
end    


% 
% %fix first day
% data= importdata('data.mat');
% data1 = data{1,1};
% final1=final_tracker(:,:,1);
% [a,b]= size(final1);
% [c,d]= size(data1)
% for i=1:a
%     for j=1:c
%     if final1(i,5)== data1(j,10) && final1(i,6)==data1(j,12) 
%        final1(i,1)=data1(j,2); 
%        final1(i,2)=data1(j,4);
%        final1(i,3)=data1(j,6); 
%        final1(i,4)=data1(j,8);
%     end
%     end
% end
% 
% final_tracker(:,:,1)= final1;

cd(strcat(HOME,'/output/tracked'))
save('tracker_meters','final_tracker');


%% CONVERT
%%% converts x- and y-coordinate pixel data to latitude and longitude data

clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

cd(strcat(HOME,'/output/tracked'))
final_tracker = importdata('tracker_meters.mat');  
[a,b,c]=size(final_tracker);


%% obtain coordinate data from worldview image
info_region= geotiffinfo(strcat(HOME,'/input/info/Land.tif')); 
im= imread(strcat(HOME,'/input/info/Land.tif'));

info_region.BoundingBox 
x_region=[info_region.BoundingBox(1)/1:info_region.RefMatrix(2)/1:info_region.BoundingBox(2)/1];  %before I divided 
y_region=[info_region.BoundingBox(4)/1:info_region.RefMatrix(4)/1:info_region.BoundingBox(3)/1];  %by 1000 now 1
 
y1_region=(y_region(1:end-1)+y_region(2:end))/2;
x1_region=(x_region(1:end-1)+x_region(2:end))/2;
 
[X_region,Y_region]=meshgrid(x1_region,y1_region);

cd(strcat(HOME,'/input/info'))
[lat,long]=polarstereo_inv(X_region,Y_region,6378273,0.081816153,70,-45);


%% convert centroid coordinates to latitude and longitude 

for k=1:c
    for i=1:a
        if final_tracker(i,5,k)~=0
            final_tracker(i,7,k) = lat(round(final_tracker(i,6,k)), round(final_tracker(i,5,k)));
            final_tracker(i,8,k) = long(round(final_tracker(i,6,k)), round(final_tracker(i,5,k)));
        else
        end
    end
end    

%save data
cd(strcat(HOME,'/output/tracked'))
save('final_tracker','final_tracker');


disp('FINISHED ONE-DAY SKIP')
disp('BEGINING TRAJECTORIES')

%% PRE-PLOT I
%Creates blank images with coast 

clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all
%% make coast images for trajectories
cd(strcat(HOME,'/output/tracked'))
order = importdata('order.mat');  

cd(strcat(HOME,'/output/BW_nice'))  %just to get the name
im_files = dir('*.tif');

for z = order
rect = [2000,3600,1000,1700];                 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
crop_number=1;                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
cd(strcat(HOME,'/output/BW_nice')) %just to get the name
fnm = im_files(z).name;
cd(strcat(HOME,'/input/info'));
bw_whole = imread('black_raw.tif');
grey_whole = imread('grey_raw.tif');


cd(strcat(HOME,'/input/images'))
%bw = imcrop(bw_whole,rect(crop_number,:));     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
%grey = imcrop(grey_whole,rect(crop_number,:)); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
grey = grey_whole;                              %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!
bw = bw_whole;                                  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!

cd(strcat(HOME,'/output/raw_black'))
imwrite(bw,fullfile(fnm),'tiff', 'Compression','none')  

%create white images
bw1= imcomplement(bw);
cd(strcat(HOME,'/output/raw_white'))
imwrite(bw1,fullfile(fnm),'tiff', 'Compression','none')  

cd(strcat(HOME,'/output/raw_grey'))
imwrite(grey,fullfile(fnm),'tiff', 'Compression','none')  
end


%%% plots the comparison for all the interpolations, filters and not post
%%% processed data


%% PLOT VELOCITIES I
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

cd(strcat(HOME,'/output/masked'))
im_files = dir('*.tif');

%title date
for i=1:size(im_files,1)/2
m{1,i}=datedisp(BEGINNING+i);
end
m = repelem(m,2);


%% get x and y data from final tracker
cd(strcat(HOME,'/output/tracked'))
final_tracker = importdata('final_tracker.mat');  
delta_t = importdata('delta_t.mat');  

[a,b,c] = size(final_tracker);
 
for k=1:c
    for i=1:a
        if final_tracker(i,1,k)~=0
        x_coord = final_tracker(i,5,k);
        y_coord = final_tracker(i,6,k);
        x_1(i,k)=x_coord; 
        y_1(i,k)=y_coord;        
        end
    end
end



%gets rid of aqua/terra image that shows up once with same time stamp
for r=1:size(x_1,1)
    s = nonzeros(x_1(r,:));
    if size(s,1)<=1
    x_1(r,:)=0;
    y_1(r,:)=0;
    end
end


cd(strcat(HOME,'/output/tracked'))
save('x_1','x_1');
save('y_1','y_1');



[a,b]= size(x_1);


% area[km^2], perimeter[km], maj[km], min[km], x[pix],y[pix], lat, long, dist[km], u[km/day],v[km/day] 
cd(strcat(HOME,'/output/tracked'))
x=x_1;
y=y_1;

[c,d] = size(x);

x_vel=x;
y_vel=y;

% badly_stiched= [4,8,16,25,58,89];
% for i=badly_stiched
%     x_vel(:,i)=zeros([c,1]);
%     y_vel(:,i)=zeros([c,1]);
% end    

%account for 1 day-skip 
%% velocity data [cm/s]
cd(strcat(HOME,'/output/raw_white'))
im_files = dir('*.tif');

u_skip=[];
v_skip=[];

for i=1:c
    if any(x_vel(i,:))
       day=find(x_vel(i,:)); 
       [~,spaces]=size(day);
       for f= 1:spaces-1
           if (day(f+1)-day(f))-1==3       %3 spaces
              delta_time=delta_t(day(f)+3) + delta_t(day(f)+2) + ... 
              delta_t(day(f)+1) + delta_t(day(f));
              u_skip(i,day(f)) = ((x_vel(i,day(f+1))-x_vel(i,day(f)))*0.250*100000/60)/(delta_time);   %[km/d]-> [cm/s]
              v_skip(i,day(f)) = ((y_vel(i,day(f+1))-y_vel(i,day(f)))*0.250*100000/60)/(delta_time);  
 
           elseif (day(f+1)-day(f))-1==2   %2 spaces
              delta_time=delta_t(day(f)+2) + delta_t(day(f)+1) + ... 
              delta_t(day(f));
              u_skip(i,day(f)) = ((x_vel(i,day(f+1))-x_vel(i,day(f)))*0.250*100000/60)/(delta_time);
              v_skip(i,day(f)) = ((y_vel(i,day(f+1))-y_vel(i,day(f)))*0.250*100000/60)/(delta_time);  

           elseif (day(f+1)-day(f))-1==1   %1 spaces
              delta_time=delta_t(day(f)+1) + delta_t(day(f));
              u_skip(i,day(f)) = ((x_vel(i,day(f+1))-x_vel(i,day(f)))*0.250*100000/60)/(delta_time);
              v_skip(i,day(f)) = ((y_vel(i,day(f+1))-y_vel(i,day(f)))*0.250*100000/60)/(delta_time);  
              
           elseif (day(f+1)-day(f))-1==0   %0 spaces
              delta_time=delta_t(day(f));
              u_skip(i,day(f)) = ((x_vel(i,day(f+1))-x_vel(i,day(f)))*0.250*100000/60)/(delta_time);
              v_skip(i,day(f)) = ((y_vel(i,day(f+1))-y_vel(i,day(f)))*0.250*100000/60)/(delta_time);    
           end      
       end 
    end
end     
        

cd(strcat(HOME,'/output/tracked'))
save('x_vel','x_vel');
save('y_vel','y_vel');
save('u','u_skip');
save('v','v_skip');





%tests guesses only
x2=x;
y2=y;
u2=u_skip;
v2=v_skip;

INDEX1=0;
%% get rid of trajectories that have too many guesses, check velocities surrounding
for i=1:c                                              %day
    if any(x(i,:)) && nnz(x(i,:))>4                    %more than 4 trajectories
       day=find(x(i,:));                               %find nonzero elements
       [~,spaces]=size(day);                           %find spaces in between them
       guess=diff(day)-1;                              %number of guesses
       guess1=diff(day);
       DAYS=day(find(guess1~=1)+1);
       occurrence=tabulate(guess);
       A=occurrence(find(occurrence(:,1)==0),3);       %percentage of guesses
       B=sum(occurrence(find(occurrence(:,1)~=0),3));  %percentage of not guesses
       if A<B                                          %if more guesses than actual trajectories                        
         point_guess=[];
         for h=2:size(DAYS,2) 
         u_ref=[];
         v_ref=[];
         guesses=[];
         vel_ref=[];
         point_guess=[x(i,DAYS(h)),y(i,DAYS(h))];      %floe to test
         idx=1;
         comp=find(DAYS(h)==day);
         for j=1:c
           point_ref=[x(j,day(comp)-1),y(j,day(comp)-1)];  %find floe near test floe on day before
           if all(point_ref) && abs(pdist2(point_ref,point_guess))<120 ...
           && u_skip(j,day(comp)-1)~=0        
           u_ref(idx,1)=u_skip(j,day(comp)-1);
           v_ref(idx,1)=v_skip(j,day(comp)-1);
           guesses(idx,1)=j;
           idx=idx+1; 
           end
         end  
         vel_ref=[mean(u_ref),mean(v_ref)];
         negative = @(val) val < 0 ;
         positive = @(val) val > 0 ;
         if all(isnan(vel_ref))                  %if no neighbors to compare, skip
            continue
         elseif ( negative(vel_ref(1,2)) && negative(v2(i,DAYS(h-1))) ) ||... %test if previous known vel makes sense
                ( positive(vel_ref(1,2)) && positive(v2(i,DAYS(h-1))) )       %same south/north direction
            continue
         else
         x2(i,DAYS(h):b)=0;
         x_vel(i,DAYS(h):b)=0;
         y2(i,DAYS(h):b)=0;
         y_vel(i,DAYS(h):b)=0;
         u2(i,DAYS(h):b)=0;
         v2(i,DAYS(h):b)=0;                   %delete otherwise
         INDEX1=INDEX1+1;
         end    
         end 
       end  
    end
end



idx=0;
%&& x(j,DAYS(h))~=0
%gets rid of aqua/terra image that shows up once with same time stamp
for r=1:size(x_1,1)
    s = nonzeros(x2(r,:));
    if size(s,1)<=1
    x2(r,:)=0;
    y2(r,:)=0;
    idx=idx+1;
    end
end



cd(strcat(HOME,'/output/tracked'))
save(['x3'],'x2');
save(['y3'],'y2');
save(['x_vel'],'x_vel');
save(['y_vel'],'y_vel');
save(['u2'],'u2');
save(['v2'],'v2');

x=x_vel;
y=y_vel;
u=u2;
v=v2;


%find outliers
ind=1;
for i=1:size(u,2)-1
for l=1:size(u,1)
     if u(l,i)~=0 
        UUU(ind,1)=u(l,i);
        VVV(ind,1)=v(l,i);
        ind=ind+1;
    end   
end
end
std_u=std(UUU);
std_v=std(VVV);
mean(UUU)
mean(VVV)

scatter (UUU,VVV)
xlim([-300 200])
ylim([-300 200])

% print images
cd(strcat(HOME,'/output/tracked'))
print('-dtiff','-r250', 'outliers1.tif')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PIV postprocessing loop
cd(strcat(HOME,'/input/info'))
amount=b;
typevector=x; %typevector will be 1 for regular vectors, 0 for masked areas

% Settings
umin =nanmean(UUU)-2*nanstd(UUU); % minimum allowed u velocity
umax =nanmean(UUU)+2*nanstd(UUU); % maximum allowed u velocity
vmin =-nanmean(VVV)-2*nanstd(VVV); % minimum allowed v velocity
vmax =-nanmean(VVV)+2*nanstd(VVV); % maximum allowed v velocity
stdthresh=3; % threshold for standard deviation check
epsilon=0.15; % epsilon for normalized median test
thresh=3; % threshold for normalized median test

u_filt=[];
v_filt=u_filt;
typevector_filt=u_filt;
for PIVresult=1:size(u,2)-1
    u_filtered=u(:,PIVresult);
    v_filtered=v(:,PIVresult);
    typevector_filtered=typevector(:,PIVresult);
    %vellimit check
    u_filtered(u_filtered<umin)=NaN;
    u_filtered(u_filtered>umax)=NaN;
    v_filtered(v_filtered<vmin)=NaN;
    v_filtered(v_filtered>vmax)=NaN;
% %     % stddev check
%     meanu=nanmean(nanmean(u_filtered));
%     meanv=nanmean(nanmean(v_filtered));
%     std2u=nanstd(reshape(u_filtered,size(u_filtered,1)*size(u_filtered,2),1));
%     std2v=nanstd(reshape(v_filtered,size(v_filtered,1)*size(v_filtered,2),1));
%     minvalu=meanu-stdthresh*std2u;
%     maxvalu=meanu+stdthresh*std2u;
%     minvalv=meanv-stdthresh*std2v;
%     maxvalv=meanv+stdthresh*std2v;
%     u_filtered(u_filtered<minvalu)=NaN;
%     u_filtered(u_filtered>maxvalu)=NaN;
%     v_filtered(v_filtered<minvalv)=NaN;
%     v_filtered(v_filtered>maxvalv)=NaN;
    % normalized median check
    %Westerweel & Scarano (2005): Universal Outlier detection for PIV data
    [J,I]=size(u_filtered);
    medianres=zeros(J,I);
    normfluct=zeros(J,I,2);
    b=1;
    for c=1:2
        if c==1; velcomp=u_filtered;else;velcomp=v_filtered;end %#ok<*NOSEM>
        for i=1+b:I-b
            for j=1+b:J-b
                neigh=velcomp(j-b:j+b,i-b:i+b);
                neighcol=neigh(:);
                neighcol2=[neighcol(1:(2*b+1)*b+b);neighcol((2*b+1)*b+b+2:end)];
                med=median(neighcol2);
                fluct=velcomp(j,i)-med;
                res=neighcol2-med;
                medianres=median(abs(res));
                normfluct(j,i,c)=abs(fluct/(medianres+epsilon));
            end
        end
    end
    info1=(sqrt(normfluct(:,:,1).^2+normfluct(:,:,2).^2)>thresh);
    u_filtered(info1==1)=NaN;
    v_filtered(info1==1)=NaN;

    x_vel(isnan(v_filtered),PIVresult)=0;
    y_vel(isnan(v_filtered),PIVresult)=0;
    typevector_filtered(isnan(u_filtered))=2;
    typevector_filtered(isnan(v_filtered))=2;
    typevector_filtered(typevector(:,PIVresult)==0)=0; %restores typevector for mask
    ufilt_nointerp(:,PIVresult)=u_filtered;     
    vfilt_nointerp(:,PIVresult)=v_filtered;     
    
    %Interpolate missing data
%     u_filtered=inpaint_nans(u_filtered,5);
%     v_filtered=inpaint_nans(v_filtered,5);
%     
%     u_filt(:,PIVresult)=u_filtered;
%     v_filt(:,PIVresult)=v_filtered;
%     typevector_filt(:,PIVresult)=typevector_filtered;
end
%disp('DONE.')

cd(strcat(HOME,'/output/tracked'))
save('u_filt','ufilt_nointerp');
save('v_filt','vfilt_nointerp');
save('x_filt','x')
save('y_filt','y')
% 
% 
% x_vel=x;
% y_vel=y;
% 
% 
% [c,d] = size(x_vel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% PRE-PLOT II 
%creates images with ice floes that are tracked only to later plot their
%trajectories
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

%% obtain centroid data
cd(strcat(HOME,'/output/tracked'))
final_tracker = importdata('final_tracker.mat');  
x = importdata('x3.mat');
y = importdata('y3.mat');
FLOE_LIBRARY=importdata('FLOE_LIBRARY.mat');


cd(strcat(HOME,'/output/tracked'))
properties = importdata('prop.mat');  
order = importdata('order.mat'); 
[a,b,c] = size(final_tracker);


cd(strcat(HOME,'/output/BW_nice'))
im_files = dir('*.tif');

cent_x=[];
cent_y=[];
[d,~]=size(x);
prop=[];
INDEX=0;

for i = order
INDEX=INDEX+1;    
cd(strcat(HOME,'/output/raw_black'))
fnm = im_files(i).name; %call image to get icefloe image
im = imread(fnm);

cd(strcat(HOME,'/output/raw_grey'))
im_grey = imread(fnm);
imgrey1=im_grey(:,:,1);
imgrey2=im_grey(:,:,2);
imgrey3=im_grey(:,:,3);

[p,q]=size(im);
prop=properties{1,INDEX};
%figure, imshow(im1)

for j=1:size(x,1)
    if x(j,INDEX)~=0 
    cent_x=x(j,INDEX); 
    cent_y=y(j,INDEX);  
    [row,col]=find(cent_x==prop(:,6));
    
    floe=[];
    floe=FLOE_LIBRARY{row,INDEX};
    Y1=prop(row,11);   
    Y2=prop(row,11)+prop(row,13);
    X1=prop(row,10);
    X2=prop(row,10)+prop(row,12);

    if Y2>p
       Y2=p;
    else 
    end

    if X2>q
       X2=q;
    else
    end

    new=zeros(p,q);    
    new(Y1:Y2, X1:X2) = floe;

    %plot on black coast image
    notzeros2 = find(new==1);
    im(notzeros2)=1;

    imgrey1(notzeros2)=255;
    imgrey2(notzeros2)=255;
    imgrey3(notzeros2)=255;
    
    im_grey_final=cat(3,imgrey1,imgrey2,imgrey3);
    
    cd(strcat(HOME,'/output/raw_grey'))
    imwrite(im_grey_final,fullfile(fnm),'tiff', 'Compression','none')   %save black image
    end  
end


cd(strcat(HOME,'/output/raw_black'))
imwrite(im,fullfile(fnm),'tiff', 'Compression','none')   %save black image
end


%% PLOT TRAJECTORIES
%Plots trajectories of ice floes with latitude and longitude on axis
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

cd(strcat(HOME,'/output/raw_grey'))
im_files = dir('*.tif');

%title date
for i=1:size(im_files,1)/2
m{1,i}=datedisp(BEGINNING+i);
end
m = repelem(m,2);

%plot
cd(strcat(HOME,'/output/tracked'))
x= importdata('x3.mat');
y= importdata('y3.mat');

cd(strcat(HOME,'/output/tracked'))
order = importdata('order.mat'); 
delta_t = importdata('delta_t.mat'); 
[c,d] = size(x);


h= ceil(c/5);
color_list1= repmat({[.61 .51 .74],[.4 .7 1], [1 .6 .8],[0 .6 0],...
    [1 .5 0],[1 .2 .2]},h);   
    %ligth purple,baby blue,pink,dark green,orange,red
list= [1,1:8];

cd(strcat(HOME,'/output/raw_grey'))
im_files = dir('*.tif');
INDEX=0;

for i = order
INDEX=INDEX+1;    
cd(strcat(HOME,'/output/raw_grey'))
fnm = im_files(i).name; %call image to plot on
im = imread(fnm);

TITLE={};
if contains(fnm,'aqua')
TITLE= strcat(m(INDEX),':  Aqua');
elseif contains(fnm,'terra')
TITLE= strcat(m(INDEX),':  Terra');
end

figure, imshow(im); hold on
    %set(gcf,'Visible','off');
    for j=1:c
        if x(j,INDEX)~=0
            [~,f]=find(x(j,1:INDEX)~=0); %f=column=3
            [~,steps]=size(f);
            if steps==1
               place=1;
            else 
                place=steps-1;
            end
            for z=1:place
            if f(1)>=INDEX
               f(z+1)=f(z);
            end   
            if f(z+1)-f(z)>1
               plot([x(j,f(z)), x(j,f(z+1))],[y(j,f(z)), y(j,f(z+1))],'.--','Color',[color_list1{1,j}],'MarkerSize',15, 'LineWidth', 2);
            else    
               plot([x(j,f(z)), x(j,f(z+1))],[y(j,f(z)), y(j,f(z+1))],'.-','Color',[color_list1{1,j}],'MarkerSize',15, 'LineWidth', 2);
            end
            end
        end
    end
    
%% axis data
[n o ~]=size(im);
xlim([0 o])
ylim([0 n])

% automatically without 0
xticks(linspace(0,o,5))
xticklabels(LIST1(1:5))

yticks(linspace(0,n,5))
yticklabels(LIST2(1:5))

set(gca,'fontsize',14)
axis on


hTitle  = title (TITLE);
hXLabel = xlabel(['\it{km} (', EXP2, ')']);
hYLabel = ylabel(['\it{km} (', EXP1, ')']);
set(gcf,'units','inches','position',[0,0,9,12])

%scale bar
rectangle('Position',[0,n-100,160,100],'FaceColor','black')
Scalebar_length = 80;   %20 km
quiver(35,n-70,Scalebar_length,0,'LineWidth',3, 'Color','white','ShowArrowHead','off','AutoScale','off')
text(25,n-30,'20 km','Color','white','FontSize',12)

%Georgy region
% hTitle  = title (TITLE);
% hXLabel = xlabel(['\it{km} (', EXP2, ')']);
% hYLabel = ylabel(['\it{km} (', EXP1, ')']);
% set(gcf,'units','inches','position',[0,0,12,8])
%
% %scalebar
% rectangle('Position',[10,n-75,100,75],'FaceColor','black')
% Scalebar_length = 80;   %40 km
% quiver(20,n-50,Scalebar_length,0,'LineWidth',3, 'Color','white','ShowArrowHead','off','AutoScale','off')
% text(20,n-20,'20 km','Color','white','FontSize',12)

cd(strcat(HOME,'/output/final'))
filename = [num2str(INDEX),'_',fnm];
print('-dtiff','-r250', filename)
hold off;
close;
end

%% PLOT TRAJECTORIES II
INDEX=0;
for i = order
INDEX=INDEX+1;    
cd(strcat(HOME,'/output/contours'))
fnm = im_files(i).name; %call image to plot on
im = imread(fnm);

TITLE={};
if contains(fnm,'aqua')
TITLE= strcat(m(INDEX),':  Aqua');
elseif contains(fnm,'terra')
TITLE= strcat(m(INDEX),':  Terra');
end

[n,o,~]= size(im);
figure, imshow(im); hold on
    %set(gcf,'Visible','off');
    for j=1:c
        if x(j,INDEX)~=0
            [~,f]=find(x(j,1:INDEX)~=0); %f=column=3
            [~,steps]=size(f);
            if steps==1
               place=1;
            else 
                place=steps-1;
            end
            for z=1:place
            if f(1)>=INDEX
               f(z+1)=f(z);
            end   
            if f(z+1)-f(z)>1
               plot([x(j,f(z)), x(j,f(z+1))],[y(j,f(z)), y(j,f(z+1))],'.--','Color',[color_list1{1,j}],'MarkerSize',15, 'LineWidth', 2);
            else    
               plot([x(j,f(z)), x(j,f(z+1))],[y(j,f(z)), y(j,f(z+1))],'.-','Color',[color_list1{1,j}],'MarkerSize',15, 'LineWidth', 2);
            end
            end
        end
    end
    
%% axis data
[n o ~]=size(im);
xlim([0 o])
ylim([0 n])

% automatically without 0
xticks(linspace(0,o,5))
xticklabels(LIST1(1:5))

yticks(linspace(0,n,5))
yticklabels(LIST2(1:5))

set(gca,'fontsize',14)
axis on


hTitle  = title (TITLE);
hXLabel = xlabel(['\it{km} (', EXP2, ')']);
hYLabel = ylabel(['\it{km} (', EXP1, ')']);
set(gcf,'units','inches','position',[0,0,9,12])

%scale bar
rectangle('Position',[0,n-100,160,100],'FaceColor','black')
Scalebar_length = 80;   %20 km
quiver(35,n-70,Scalebar_length,0,'LineWidth',3, 'Color','white','ShowArrowHead','off','AutoScale','off')
text(25,n-30,'20 km','Color','white','FontSize',12)

%Georgy region
% hTitle  = title (TITLE);
% hXLabel = xlabel(['\it{km} (', EXP2, ')']);
% hYLabel = ylabel(['\it{km} (', EXP1, ')']);
% set(gcf,'units','inches','position',[0,0,12,8])
%
% %scalebar
% rectangle('Position',[10,n-75,100,75],'FaceColor','black')
% Scalebar_length = 80;   %40 km
% quiver(20,n-50,Scalebar_length,0,'LineWidth',3, 'Color','white','ShowArrowHead','off','AutoScale','off')
% text(20,n-20,'20 km','Color','white','FontSize',12)

cd(strcat(HOME,'/output/final2'))
filename = [num2str(INDEX),'_',fnm];
print('-dtiff','-r250', filename)
hold off;
close;
end


disp('FINISHED TRAJECTORIES')
disp('BEGINING DISPERSION')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SINGLE DISPERSION
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

%NSIDC file
cd(strcat(HOME,'/input/info')) 
filename1=nsidc_file; %call image to get icefloe image
ncdisp(filename1);

lat1 =ncread(filename1,'latitude');  %[100 km] -> m
lon1 =ncread(filename1,'longitude');

velocity_u2= ncread(filename1,'u');
velocity_v2= ncread(filename1,'v');

velocity_u2(velocity_u2==-9999)=NaN;
velocity_v2(velocity_v2==-9999)=NaN;

times_icee= ncread(filename1,'time');
DateString = {'01/01/1970'};
formatIn = 'mm/dd/yyyy';
adjusted_time=datenum(DateString,formatIn)+ (times_icee);
times_ice = datetime(adjusted_time,'ConvertFrom','datenum');

%temporal restriction
for i=90:263
velocity_u1(:,:,i-89)=velocity_u2(:,:,i);
velocity_v1(:,:,i-89)=velocity_v2(:,:,i);
end

lat_reg=find(lat1>=68 & lat1<=84);
lon_reg=find(lon1>=-40 & lon1<=15);
reg=intersect(lat_reg,lon_reg);

[row2,col2]=ind2sub(size(velocity_u1(:,:,1)),reg);

row1=min(row2);
row11=max(row2);
col1=min(col2);
col11=max(col2);

%regional restriction
lat=lat1(row1:row11,col1:col11);
lon=lon1(row1:row11,col1:col11);
for rr=1:size(velocity_u2,3)
velocity_u(:,:,rr)=velocity_u2(row1:row11,col1:col11,rr);
velocity_v(:,:,rr)=velocity_v2(row1:row11,col1:col11,rr);
end


%% transform to meridional and zonal 
% formula
%E:   u * cos L  +  v * sin L
%N:  -u * sin L  +  v * cos L
for k=1:size(velocity_u,3)
for i=1:size(velocity_u,1)
for j=1:size(velocity_u,2)     
    if ~isnan(velocity_u(i,j,k)) || ~isnan(velocity_v(i,j,k))
    zonal_ice(i,j,k)=double( (velocity_u(i,j,k) * cosd(lon(i,j)) ) + (velocity_v(i,j,k) * sind(lon(i,j)) ));
    meridional_ice(i,j,k)=double((-velocity_u(i,j,k) * sind(lon(i,j)) ) + (velocity_v(i,j,k) * cosd(lon(i,j)) ));
    end
end
end
end

% %checks
% %1)at L=0o, a u-only vector (to the right) corresponds to east (positive E)
% %2)at L=0o, a v-only vector (toward the top) corresponds to north (positive N)
% %3) at L=90o, a u-only vector (to the right) corresponds to south (negative N)
% %4)at L=90o, a v-only vector (toward the top) corresponds to east (positive E)
% 
% velocity_u(i,j,k)=0;
% velocity_v(i,j,k)=10;
% lon(i,j,k)=90;
% east=double( (velocity_u(i,j,k) * cosd(lon(i,j)) ) + (velocity_v(i,j,k) * sind(lon(i,j)) ))
% north=double((-velocity_u(i,j,k) * sind(lon(i,j)) ) + (velocity_v(i,j,k) * cosd(lon(i,j)) ))

%% transform to east and north in stereographic projection
cd(strcat(HOME,'/input/info/ArcticMappingTools')) 
for k=1:size(zonal_ice,3)
[eastward_ice1,northward_ice1]=uv2vxvyn(lat,lon,zonal_ice(:,:,k),meridional_ice(:,:,k));
eastward_ice(:,:,k)=eastward_ice1;
northward_ice(:,:,k)=northward_ice1;
end

                
%dispersion
DAYS=3;
cd(strcat(HOME,'/output/tracked'))
order = importdata('order.mat'); 
sat_order = importdata('sat_order.mat'); 
delta_t = importdata('delta_t.mat');  
x0=importdata('x3.mat');
y0=importdata('y3.mat');
x= importdata('x3.mat');
y= importdata('y3.mat');



%% new grid
yq=(1450:1440:1440*size(x,2)/2);%/(60*24)
% yq(1,1)=0;
% yq(1,2)=55;
% yq(1,3:length(yq1)+2)=yq1;

%TT1=zeros([size(x,1) 12]);                                                 %%change depending on how big
%interpolate location data
for i=1:size(x,1)
if any(x(i,:))
tt1=[];
k1 = find(x(i,:),1,'first');
k2 = find(x(i,:),1,'last');

idx=1;
for ii=k1:k2-1
idx=idx+1;
tt1(1,idx)=sum(delta_t(k1:ii));
%TT1(i,1:length(tt1))=tt1;
end
tt1(1,1)=0;
if tt1(end)>150
[val_kk1,kk1]=min(abs(yq-tt1(1)));
[val_kk2,kk2]=min(abs(yq-tt1(end)));
x1=x(i,k1:k2);
x1(x1==0)=NaN;
y1=y(i,k1:k2);
y1(y1==0)=NaN;
X1=interp1(tt1(1:idx),x1,yq(kk1:kk2),'pchip');
Y1=interp1(tt1(1:idx),y1,yq(kk1:kk2),'pchip');

else
X1=0;
Y1=0;
x1=x(i,k1:k2);
x1(x1==0)=NaN;
y1=y(i,k1:k2);
y1(y1==0)=NaN;
end
XY_interp(i,2:length(X1)+1,1)=X1;                               %x_centroid
XY_interp(i,1,1)=x1(1,1);
XY_interp(i,2:length(Y1)+1,2)=Y1;                               %y_centroid
XY_interp(i,1,2)=y1(1,1);
if X1==0
  day_frame=1;
else
  day_frame=length(X1)+1;
end
XY_interp(i,1:day_frame,3)=ceil(k1/2):ceil(k2/2);            %day
%XY_interp(i,1,3)=ceil(k1/2);
end
end

count1=0;
count2=0;
%go through and make sure interpolation wasn't too extreme
for i=1:size(XY_interp,1)
test1=x(i,:);
test1=test1(test1~=0);
test1=test1(~isnan(test1));
test2=XY_interp(i,:,1);
test2=test2(test2~=0);
test2=test2(~isnan(test2));

A=max(test1)+100;
B=min(test1)-100;


oddball1=test2(test2<B);
oddball2=test2(test2>A);
if ~isempty(oddball1)
[~,numb]=find(ismember(XY_interp(i,:,1),oddball1));
XY_interp(i,numb,1:2)=NaN;
count1=count1+1;
del1(count1,1)=i;
end
if ~isempty(oddball2)
[~,numb]=find(ismember(XY_interp(i,:,1),oddball2));
XY_interp(i,numb,1:2)=NaN;
count2=count2+1;
del2(count2,1)=i;
end
end

XY_interp(isnan(XY_interp))=0;
delta_tt=yq;

AA=XY_interp(:,:,1);
BB=XY_interp(:,:,2);
CC=XY_interp(:,:,3);

ice_days=CC;

east_ice=[];
north_ice=[];
[c,~,~] = size(XY_interp);

%get velocities from these displacements
for i=1:c
    if any(XY_interp(i,:,1))
    day=find(XY_interp(i,:,1)); 
    [~,spaces]=size(day);
    for f= 1:spaces-1
        east_ice(i,day(f)) = (XY_interp(i,day(f+1),1)-XY_interp(i,day(f),1))/...
                  (delta_tt(day(f+1))-delta_tt(day(f)))*(25000/60);   %[cm/s]
        north_ice1(i,day(f)) = (XY_interp(i,day(f+1),2)-XY_interp(i,day(f),2))/...
                  (delta_tt(day(f+1))-delta_tt(day(f)))*(25000/60);   %[cm/s]
    end
    end
end
north_ice=-north_ice1;
           
cd(strcat(HOME,'/input/info/ArcticMappingTools')) 
[x_ice,y_ice]=ll2psn(lat,lon);
x_ice=double(x_ice);
y_ice=double(y_ice);

% %visualize data
% cd('/Users/rosalinda/Documents/UCR/Research/ice_tracker/branch1/ArcticMappingTools');
% %quiverpsn(lat,lon,zonal_ice(:,:,1),meridional_ice(:,:,1))
% quiver(x_ice,y_ice,eastward_ice(:,:,1),northward_ice(:,:,1));
% hold on; 
% greenland



%% buoy information
cd(strcat(HOME,'/input/info')) 
info_region= geotiffinfo('NE_Greenland.2017100.terra.250m.tif'); 

info_region.BoundingBox %this reads:[left top corner, right top corner, right bottom corner, left bottom corner]
x_region=[info_region.BoundingBox(1)/1:info_region.RefMatrix(2)/1:info_region.BoundingBox(2)/1];  %before I divided by 1000 now 1
y_region=[info_region.BoundingBox(4)/1:info_region.RefMatrix(4)/1:info_region.BoundingBox(3)/1];
 
y1_region=(y_region(1:end-1)+y_region(2:end))/2;
x1_region=(x_region(1:end-1)+x_region(2:end))/2;
 
[X_region,Y_region]=meshgrid(x1_region,y1_region);

% crop image to our area of interest
%rect = [1500,1900,1300,1700];
%rect = [2000,3600,1000,1700];
%rect = [2000,4000,1000,1700];

X_cropped = X_region;%(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3));
Y_cropped =Y_region;%(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3));

X_trans=zeros(size(AA));
Y_trans=zeros(size(AA));

for i=1:size(AA,1)
for j=1:size(AA,2)
if AA(i,j)~=0 && ~isnan(AA(i,j))    
X_trans(i,j)=(AA(i,j)*250)+X_cropped(1);
Y_trans(i,j)=Y_cropped(1)-(BB(i,j)*250);
end 
end
end


cd(strcat(HOME,'/output'))
ice_points_x=[];
ice_points_y=[];
%find in map
for i=1:size(X_trans,1)
for j=1:size(X_trans,2)
if X_trans(i,j)~=0 && ~isnan(X_trans(i,j))
pt=[X_trans(i,j),Y_trans(i,j)];
dist=zeros(size(x_ice));

for ii=1:size(x_ice,1)*size(x_ice,2)
dist(ii)=pdist2(pt,[x_ice(ii),y_ice(ii)]);
end

[found_pt1,found_pt2]=find(min(min(dist))==dist);
ice_points_x(i,j)=found_pt1;
ice_points_y(i,j)=found_pt2;
end
end
end


cd(strcat(HOME,'/output/tracked'))
save('ice_points_x','ice_points_x');
save('ice_points_y','ice_points_y');




cd(strcat(HOME,'/input/info/ArcticMappingTools'));
[list1,list2]=find(AA~=0 & ~isnan(AA));
for i=1:length(list1) 
ice_x(list1(i),list2(i))=x_ice(ice_points_x(list1(i),list2(i)),ice_points_y(list1(i),list2(i)));
ice_y(list1(i),list2(i))=y_ice(ice_points_x(list1(i),list2(i)),ice_points_y(list1(i),list2(i)));
end

%make 0's nans
for k=1:size(eastward_ice,3)
for i=1:size(eastward_ice,1)
for j=1:size(eastward_ice,2)   
    if eastward_ice(i,j,k)==0
    eastward_ice(i,j,k)=NaN;
    end
    if northward_ice(i,j,k)==0
    northward_ice(i,j,k)=NaN;
    end
end
end
end

days_list=15;
%days_list=[1,2,3,4,5,7,10,15,22,30,45,60,90,180];

for iii=1:size(days_list,2)
DAY_DISP=days_list(iii);
    
%get mean velocities for each vel w/at least 3 vectors for average
for i=1:size(eastward_ice,1)
for j=1:size(eastward_ice,2)   
for dayz=1:(size(eastward_ice,3))

   time_frame1=dayz-DAY_DISP;        %time frame to calculate mean- 10 days
   time_frame2=dayz+DAY_DISP;        %time frame to calculate mean- 10 days
   time_frame3=(time_frame1:time_frame2);
   time_frame=time_frame3(time_frame3>0 & time_frame3<size(eastward_ice,3));
   mean_map_u(i,j,dayz)=nansum(eastward_ice(i,j,time_frame))/sum(eastward_ice(i,j,time_frame)~=0 & ~isnan(eastward_ice(i,j,time_frame))); 
   mean_map_v(i,j,dayz)=nansum(northward_ice(i,j,time_frame))/sum(northward_ice(i,j,time_frame)~=0 & ~isnan(northward_ice(i,j,time_frame)));     


end
end
end

mean_map_u=double(mean_map_u);
mean_map_v=double(mean_map_v);

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % for ii=1:size(mean_map_u,3)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % if any(any(mean_map_u(:,:,ii)))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %visualize data
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % cd('/Users/rosalinda/Documents/UCR/Research/ice_tracker/branch1/ArcticMappingTools');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %quiverpsn(lat,lon,zonal_ice(:,:,1),meridional_ice(:,:,1))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % quiver(x_ice,y_ice,mean_map_u(:,:,ii),mean_map_v(:,:,ii));
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % hold on; 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % greenland
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % ylim([-2800000 0])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % xlim([0 1300000])
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % print images
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % cd(strcat(HOME,'/output/mean_field_NSIDC'))
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % filename = [num2str(ii),'_avg_days_',num2str(DAY_DISP*2+1),'.tif'];
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % print('-dtiff','-r0', filename)
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % close;
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % end




% % %%Visualize vectors
% % %mean vector 
% % quiver(X_trans(i),Y_trans(i),mean_ice_eastward_interp(i)*scale,mean_ice_northward_interp(i)*scale,'Color','red','LineWidth',2,'AutoScale','off')             
% % 
% % hold on;
% %    
% % %instantaneous vector
% % quiver(X_trans(i),Y_trans(i),east_ice(i)*scale,north_ice(i)*scale,'Color','black','LineWidth',2,'AutoScale','off')
% % 
% % %components
% % [THETA_I,R] = cart2pol(mean_ice_eastward_interp(i)*scale,mean_ice_northward_interp(i)*scale); %Convert to polar coordinates
% % 
% % %along
% % [xr,yr]=pol2cart(-THETA_I,VX_ice_eastward(i)*scale);
% % plot([X_trans(i) X_trans(i)+xr],[Y_trans(i) Y_trans(i)-yr],'--','Color',[.6 .8 1]); 
% % 
% % %across
% % THETA_II=-THETA_I+(pi/2);
% % [xxr,yyr]=pol2cart(-THETA_II,VY_ice_northward(i)*scale);
% % plot([X_trans(i)+xr X_trans(i)+xr-xxr],[Y_trans(i)-yr Y_trans(i)-yr-yyr],'--','Color',[1 0 .498]); 

%% INTERPOLATE
%%%TOO SLOW

%get mean velocities for each vel w/at least 3 vectors for average

for i=1:size(X_trans,1)
if any(X_trans(i,:))  
   spaces=find(X_trans(i,:)~=0);
   for f=spaces
   day_quest=CC(i,f);
   xqq=X_trans(i,f);
   yqq=Y_trans(i,f);
   
   idx=0;
   list_x=[];
   list_y=[];
   list_mean_map_u=[];
   list_mean_map_v=[];
   for zz=1:size(mean_map_u,1)
   for yy=1:size(mean_map_u,2)
      if ~isnan(mean_map_u(zz,yy,day_quest))          
      idx=idx+1;
      list_x(1,idx)=x_ice(zz,yy);
      list_y(1,idx)=y_ice(zz,yy);
      list_mean_map_u(1,idx)=mean_map_u(zz,yy,day_quest);
      list_mean_map_v(1,idx)=mean_map_v(zz,yy,day_quest);
      end
   end
   end


   %[xq,yq] = meshgrid(AA(i,f):1:AA(i,f)+1,BB(i,f):1:BB(i,f)+1)
   cd(strcat(HOME,'/input/info')) 
   uq = griddata(list_x,list_y,list_mean_map_u,xqq,yqq,'v4');
   vq = griddata(list_x,list_y,list_mean_map_v,xqq,yqq,'v4');
    
   mean_ice_eastward_interp(i,f)=uq;
   mean_ice_northward_interp(i,f)=vq;     
   end
end
end




%scatter(xqq,yqq)


% % %get mean velocities for each vel w/at least 3 vectors for average 
% % for dayz=1:(size(eastward_ice,3))   
% %    time_frame1=dayz-DAY_DISP;        %time frame to calculate mean- 10 days
% %    time_frame2=dayz+DAY_DISP;        %time frame to calculate mean- 10 days
% %    time_frame3=(time_frame1:time_frame2);
% %    time_frame=time_frame3(time_frame3>0 & time_frame3<size(eastward_ice,3));
% % 
% %    [x_idx,y_idx]=find(ismember(CC,time_frame));
% % 
% %    for gg=1:size(x_idx,1)
% %    xqq(gg)=X_trans(x_idx(gg),y_idx(gg));
% %    yqq(gg)=Y_trans(x_idx(gg),y_idx(gg));
% %    end
% %     
% %    mean_map_u2=mean_map_u(:,:,dayz);
% %    mean_map_u_A=mean_map_u2(:);
% %    mean_map_u_B=mean_map_u_A;
% %    mean_map_u_A(isnan(mean_map_u_B))=[];
% %    
% %    
% %    mean_map_v2=mean_map_v(:,:,dayz);
% %    mean_map_v_A=mean_map_v2(:);
% %    mean_map_v_A(isnan(mean_map_u_B))=[];
% % 
% %    
% %    x_ice_A=x_ice(:);
% %    y_ice_A=y_ice(:);
% %    
% %    x_ice_A(isnan(mean_map_u_B))=[];
% %    y_ice_A(isnan(mean_map_u_B))=[];
% %    
% %    
% %    uq = griddata(x_ice_A,y_ice_A,mean_map_u_A,xqq,yqq,'cubic');
% %    vq = griddata(x_ice_A,y_ice_A,mean_map_v_A,xqq,yqq,'cubic');
% %    
% %    
% % %    for TEST=1:size(xq,2)
% % %    uq(1,TEST) = griddata(x_ice_A,y_ice_A,mean_map_u_A,xq(TEST),yq(TEST),'cubic');
% % %    vq(1,TEST) = griddata(x_ice_A,y_ice_A,mean_map_v_A,xq(TEST),yq(TEST),'cubic');
% % %    end
% %    
% %    for gg=1:size(x_idx,1)
% %    mean_ice_eastward_interp(x_idx(gg),y_idx(gg))=uq(1,gg);
% %    mean_ice_northward_interp(x_idx(gg),y_idx(gg))=vq(1,gg);  
% %    end
% %    
% % end

%get velocity components from mean velocities 
for i=1:size(east_ice,1)
if any(east_ice(i,:))  
   day=find(east_ice(i,:)); 
   [~,spaces]=size(day);
   for f= 1:spaces

% %example        
% mean_u(i,day(f))=-5
% mean_v(i,day(f))=-3
% u(i,day(f))=5
% v(i,day(f))=3

   A=sqrt(east_ice(i,day(f))^2+north_ice(i,day(f))^2);
   B=sqrt(mean_ice_eastward_interp(i,day(f))^2+mean_ice_northward_interp(i,day(f))^2);
   
   if mean_ice_eastward_interp(i,day(f))==0  || mean_ice_northward_interp(i,day(f))==0  || ...
      ( mean_ice_eastward_interp(i,day(f))==east_ice(i,day(f)) && mean_ice_northward_interp(i,day(f))==north_ice(i,day(f)) ) || ...
      ~isreal(acosd ( ( (mean_ice_eastward_interp(i,day(f))*east_ice(i,day(f))) + ...
         (mean_ice_northward_interp(i,day(f))*north_ice(i,day(f))) )  / (A*B) ))
   VX_ice_eastward(i,day(f))=NaN;
   VX_ice_eastward(i,day(f))=NaN;
   else    
   THETA=acosd ( ( (mean_ice_eastward_interp(i,day(f))*east_ice(i,day(f))) + ...
         (mean_ice_northward_interp(i,day(f))*north_ice(i,day(f))) )  / (A*B) ); 
         
   R = [cosd(90) -sind(90); sind(90) cosd(90)]; %rotate CCW 90deg
   point1 = [mean_ice_eastward_interp(i,day(f)) mean_ice_northward_interp(i,day(f))]'; % arbitrarily selected
   rotpoint1 = R*point1;      %y
   rotpoint2 = R*rotpoint1;   %x
   rotpoint3 = R*rotpoint2;   %y
   rotpoint4 = R*rotpoint3;   %x
   
   point2 = [east_ice(i,day(f)) north_ice(i,day(f))]';
   
   %get direction
   test1(1,1)=point1(1)+.1;  %right
   test1(2,1)=point1(2);
   
   d_test=((test1(1) -rotpoint4(1))*(rotpoint4(2)-rotpoint2(2))) -...   %x line
   ((test1(2) -rotpoint4(2))*(rotpoint4(1)-rotpoint2(1)));
   x_dir_test=d_test/abs(d_test);
   
   d=((point2(1) -rotpoint4(1))*(rotpoint4(2)-rotpoint2(2))) -...   %x line
   ((point2(2) -rotpoint4(2))*(rotpoint4(1)-rotpoint2(1)));
   x_dir=d/abs(d);
     
   %get direction
   test2(1,1)=rotpoint1(1);  
   test2(2,1)=rotpoint1(2)+.1;  %up
   
   e_test=((test2(1) -rotpoint1(1))*(rotpoint1(2)-rotpoint3(2))) -...   %y line
   ((test2(2) -rotpoint1(2))*(rotpoint1(1)-rotpoint3(1)));
   y_dir_test=e_test/abs(e_test);
   
   
   e=((point2(1) -rotpoint1(1))*(rotpoint1(2)-rotpoint3(2))) -...   %y line
   ((point2(2) -rotpoint1(2))*(rotpoint1(1)-rotpoint3(1)));
   y_dir=e/abs(e);
   
   theta_big=0;   
   %get direction of vectors    
   if THETA>90
      theta_big=1;
      THETA=180-THETA;
   end
   
   VX1=A*cosd(THETA);   %we are interested on across mean vel drift
   VY1=A*sind(THETA);   %mean vel is x axis, thus we need y_dir drift

   switch true 
   case y_dir>0 && x_dir<0
   quad=1;
   VX_ice_eastward(i,day(f))=VX1;                                            %[cm/s]
   VY_ice_northward(i,day(f))=VY1;                                            %[cm/s]
   case y_dir<0 && x_dir<0
   quad=2;
   VX_ice_eastward(i,day(f))=-VX1;
   VY_ice_northward(i,day(f))=VY1;  
   case y_dir<0 && x_dir>0
   quad=3;
   VX_ice_eastward(i,day(f))=-VX1;
   VY_ice_northward(i,day(f))=-VY1;  
   case y_dir>0 && x_dir>0
   quad=4;
   VX_ice_eastward(i,day(f))=VX1;
   VY_ice_northward(i,day(f))=-VY1;  
   otherwise 
   quad=0;
   VY_ice_northward(i,day(f))=VY1;   
   if theta_big==1
   VX_ice_eastward(i,day(f))=-VX1;
   else
   VX_ice_eastward(i,day(f))=VX1;
   end
   end
   end
   end   
end
end

%get rid of imaginary numbers
ff=zeros(size(VY_ice_northward));
for ii=1:size(VY_ice_northward)
if ~isreal(VY_ice_northward(ii))
ff(ii)=1;
end
end


for zz=1:size(VX_ice_eastward,1)
   day_inquest=find(VX_ice_eastward(zz,:));
   for gg=1:size(day_inquest,2)
   if day_inquest(gg)==1 && VX_ice_eastward(zz,gg)~=0
      days1(zz,day_inquest(gg))=1450;
   else
      days1(zz,day_inquest(gg))=1440;
   end
   end
end


%% Displacement plot
for i=1:size(VX_ice_eastward,1)
if any(VX_ice_eastward(i,:))  
   day=find(VX_ice_eastward(i,:)); 
   [~,spaces]=size(day);
   for f= 1:spaces
   %now dispersion dy=vdt
   DX_ice_eastward1(i,day(f))=VX_ice_eastward(i,day(f))*days1(i,day(f),1)*(60/100000);   %[km] ALONG
   DY_ice_northward1(i,day(f))=VY_ice_northward(i,day(f))*days1(i,day(f),1)*(60/100000);   %[km] ACROSS
   end
end
end

for i=1:size(DX_ice_eastward1,1)
if any(DX_ice_eastward1(i,:))  
   day=find(DX_ice_eastward1(i,:)); 
   [~,spaces]=size(day);
   for f= 1:spaces
   DX_ice_eastward(i,day(f))=sum(DX_ice_eastward1(i,day(1):day(f)));     %[km] ALONG
   DY_ice_northward(i,day(f))=sum(DY_ice_northward1(i,day(1):day(f)));   %[km] ACROSS
   end
end
end

%displacement plot
figure,
for i=1:size(DY_ice_northward,1)
if any(DY_ice_northward(i,:))
plot([0 delta_tt(1,find(DY_ice_northward(i,:)))]/(60*24),[0 DY_ice_northward(i,find(DY_ice_northward(i,:)))])
hold on;
%pause;
end
end
title ('Displacement Fluctuations','interpreter','latex','FontSize',12)
ylabel('Disp fluct across \it[km]','interpreter','latex','FontSize',10);
xlabel ('time \it[days]','interpreter','latex','FontSize',10)
cd(strcat(HOME,'/output/single_particle/plots_NSIDC'))
filename = ['displacement_',num2str(DAY_DISP*2+1),'.tif'];
print('-dtiff','-r0', filename)
hold off
close;


for r=[1:20:500]
th = 0:pi/50:2*pi;
xunit = r * cos(th);
yunit = r * sin(th);
plot(xunit, yunit,'Color','red');
hold on
end

%displacement map
for i=1:size(DY_ice_northward,1)
if any(DY_ice_northward(i,:))
plot([0 DX_ice_eastward(i,find(DY_ice_northward(i,:)))],[0 DY_ice_northward(i,find(DY_ice_northward(i,:)))])
hold on;
%pause;
end
end

title ('Displacement Fluctuations','interpreter','latex','FontSize',12)
ylabel('y \it[km]','interpreter','latex','FontSize',10);
xlabel ('x \it[km]','interpreter','latex','FontSize',10);
ylim([-300 300])
xlim([-500 500])
cd(strcat(HOME,'/output/single_particle/plots_NSIDC'))
filename = ['radius_displacement_',num2str(DAY_DISP*2+1),'.tif'];
print('-dtiff','-r0', filename)
hold off
close;



%% buoy information
cd(strcat(HOME,'/input/info')) 
info_region= geotiffinfo('NE_Greenland.2017100.terra.250m.tif'); 

info_region.BoundingBox %this reads:[left top corner, right top corner, right bottom corner, left bottom corner]
x_region=[info_region.BoundingBox(1)/1:info_region.RefMatrix(2)/1:info_region.BoundingBox(2)/1];  %before I divided by 1000 now 1
y_region=[info_region.BoundingBox(4)/1:info_region.RefMatrix(4)/1:info_region.BoundingBox(3)/1];
 
y1_region=(y_region(1:end-1)+y_region(2:end))/2;
x1_region=(x_region(1:end-1)+x_region(2:end))/2;
 
[X_region,Y_region]=meshgrid(x1_region,y1_region);

% crop image to our area of interest
%rect = [1500,1900,1300,1700];
%rect = [2000,3600,1000,1700];
%rect = [2000,4000,1000,1700];

X_cropped = X_region;%(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3));
Y_cropped =Y_region;%(rect(2):rect(2)+rect(4), rect(1):rect(1)+rect(3));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Autocorrelation function with built in function
%across stream
ACF2_u=[];
acf2_u=zeros([size(VX_ice_eastward,1) size(VX_ice_eastward,2)-1]);
for i=1:size(VX_ice_eastward,1)
if any(VX_ice_eastward(i,:))    
    
N=find(VX_ice_eastward(i,:),1,'first');
M=find(VX_ice_eastward(i,:),1,'last');

[correlation,lags]=xcorr(VX_ice_eastward(i,N:M),'coef');  %calcuates the autocorrelation

CORR=correlation(find(lags>=0));
acf2_u(i,1:size(CORR,2)) = CORR;
end
end

%delete zero rows
acf22_u=acf2_u;
acf2_u(~any(acf22_u,2),:) = [];

%take mean for function
for j=1:size(acf2_u,2)
ACF2_u(1,j)= sum(acf2_u(:,j),1) ./ sum(acf2_u(:,j)~=0,1);
end

ACF22_u=ACF2_u;
ACF22_u(isnan(ACF2_u))=0;



%along stream
ACF2_v=[];
acf2_v=zeros([1000 size(VY_ice_northward,2)-1]);
for i=1:size(VY_ice_northward,1)
if any(VY_ice_northward(i,:))    
    
N=find(VY_ice_northward(i,:),1,'first');
M=find(VY_ice_northward(i,:),1,'last');

[correlation lags]=xcorr(VY_ice_northward(i,N:M),'coef');  %calcuates norm autocorrelation
CORR=correlation(find(lags>=0));
acf2_v(i,1:size(CORR,2)) = CORR;


[correlation,lags]=xcorr(VY_ice_northward(i,N:M));         %calcuates autocorrelation
CORR=correlation(find(lags>=0));
acf2_v_non(i,1:size(CORR,2)) = CORR;
end
end

%normalized
%delete zero rows
acf22_v=acf2_v;
acf2_v(~any(acf22_v,2),:) = [];
%take mean for function
for j=1:size(acf2_v,2)
ACF2_v(1,j)= sum(acf2_v(:,j),1) ./ sum(acf2_v(:,j)~=0,1);
end
ACF22_v=ACF2_v;
ACF22_v(isnan(ACF2_v))=0;



%not normalized
%delete zero rows
acf22_v_non=acf2_v_non;
acf2_v_non(~any(acf22_v_non,2),:) = [];
%take mean for function
for j=1:size(acf2_v_non,2)
ACF2_v_non(1,j)= sum(acf2_v_non(:,j),1) ./ sum(acf2_v_non(:,j)~=0,1);
end
ACF22_v_non=ACF2_v_non;
ACF22_v_non(isnan(ACF2_v_non))=0;




LAST1=find(ACF22_u,1,'last');
LAST2=find(ACF22_v,1,'last');
LAST=max([LAST1;LAST2]);
figure,
%plot
plot(1:size(ACF2_u,2),ACF2_u, 'LineWidth',3)
hold on;

plot(1:size(ACF2_v,2),ACF2_v,'LineWidth',3)
xlim([1 LAST])
title ('Autocorrelation function','interpreter','latex','FontSize',12)
ylabel('$\chi(\tau)$','interpreter','latex','FontSize',10);
xlabel ('time lags \it(d)','interpreter','latex','FontSize',10)
legend('along','across')
set(gca,'fontsize',18)
xlim([1 20])
cd(strcat(HOME,'/output/single_particle/plots_NSIDC'))
filename = ['auto_corr_',num2str(DAY_DISP*2+1),'.tif'];
print('-dtiff','-r0', filename)
hold off
close;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Integral time scales
%along
% limit_u=min(find(ACF22_u<0));
% m=ACF22_u(limit_u)-ACF22_u(limit_u-1)/(limit_u-(limit_u-1));
% b=-((limit_u)*m)+ACF22_u(limit_u);
% cross=-b/m;
% time1=trapz(1:limit_u-1,ACF22_u(1:limit_u-1));
% time2=trapz([limit_u-1 cross],[ACF22_u(limit_u-1) 0]);
% Int_time_along = time1 + time2;                                          %ALONG

%across
% limit_v=min(find(ACF22_v<0));
% m=ACF22_v(limit_v)-ACF22_v(limit_v-1)/(limit_v-(limit_v-1));
% b=-((limit_v)*m)+ACF22_v(limit_v);
% cross=-b/m;
% time1=trapz(1:limit_v-1,ACF22_v(1:limit_v-1));
% time2=trapz([limit_v-1 cross],[ACF22_v(limit_v-1) 0]);
% Int_time_across = time1 + time2;                                          %ACROSS
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Absolute dispersion
% get dipsersion along and across mean velocity
disp_along=[];
disp_across=[];

%arrange them by the summation of the time
%calculate dispersion
for i=1:size(DX_ice_eastward,1)
if any(DX_ice_eastward(i,:)) && sum(DX_ice_eastward(i,:)~=0)>=DAYS-1        %X days trajectory length
   day=find(DX_ice_eastward(i,:)); 
   [~,spaces]=size(day);
    for f= 1:spaces   
    disp_along(i,f) = DX_ice_eastward(i,day(f)).^2;                   %[km^2]
    disp_across(i,f) = DY_ice_northward(i,day(f)).^2;  
    end
end
end


%delete zero rows
disp_along1=disp_along;
disp_along(~any(disp_along1,2),:) = [];

disp_across1=disp_across;
disp_across(~any(disp_across1,2),:) = [];
   
disp_along((disp_along==0))=NaN;
disp_across((disp_across==0))=NaN;

N_error=sum(~isnan(disp_along));
error_bar_across=nanstd(disp_across1)./sqrt(N_error);
error_bar_along=nanstd(disp_along1)./sqrt(N_error);

abs_disp_across=nanmean(disp_across);
abs_disp_along=nanmean(disp_along);  

abs_disp_across(isnan(abs_disp_across))=0;
abs_disp_along(isnan(abs_disp_along))=0;


% loglog(delta_tt(:,1:size(abs_disp_along,2))/(60*24),abs_disp_along) %[mins]->[days]
% hold on
% loglog(delta_tt(:,1:size(abs_disp_across,2))/(60*24),abs_disp_across)
% hold on
% loglog(1:25,1:25)
% loglog(1:25,(1:25).^2)
% loglog(1:25,(1:25).^(5/4))



%loglog(delta_tt(:,1:20)/(60*24),abs_disp_along(1:20),'LineWidth',3) %[mins]->[days]
%hold on
%loglog(delta_tt(:,1:20)/(60*24),abs_disp_across(1:20),'LineWidth',3)

final_abs_disp_along=abs_disp_along(1:20);
final_abs_disp_across=abs_disp_across(1:20);
final_delta_tt=delta_tt(:,1:20)/(60*24);


%% Fit: 'untitled fit 1'.
[xData, yData] = prepareCurveData( final_delta_tt, final_abs_disp_across );

% Set up fittype and options.
ft = fittype( 'power1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.StartPoint = [71.7494403243912 1.5137025059178];

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );
% Plot fit with data.
plot( fitresult, xData, yData);
set(gca, 'YScale', 'log')
set(gca, 'XScale', 'log')

hold on;

Power_coeff=coeffvalues(fitresult);
A1=Power_coeff(1);
B1=Power_coeff(2);




loglog(final_delta_tt,final_abs_disp_along,'LineWidth',3) %[mins]->[days]
hold on
loglog(final_delta_tt,final_abs_disp_across,'LineWidth',3)
hold on
ylim([1 100000])

hold on
loglog(1:10,1:10)
loglog(1:10,(1:10).^2)
loglog(1:10,(1:10).^(5/4))

title ('Absolute Dispersion (fitted line=ax$^b$)','interpreter','latex','FontSize',20)
xlabel('time \it(d)','interpreter','latex','FontSize',10);
ylabel ('$< {r}^2_i> \it(km^2) $','interpreter','latex','FontSize',10)
legend(strcat('a=',num2str(A1),'; b=',num2str(B1)),'fitted curve','along','across','1','2','5/4','Location','northwest')
set(gca,'fontsize',14)
%axis on


cd(strcat(HOME,'/output/single_particle/plots_NSIDC'))
filename = ['abs_dispersion_',num2str(DAY_DISP*2+1),'.tif'];
print('-dtiff','-r0', filename)
hold off
close;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PDF of vel fluctuations ALONG
VX_ice_eastward(VX_ice_eastward==0)=NaN;
histogram(VX_ice_eastward,'Normalization','pdf'); %plot estimated pdf from the generated data

numBins=50; %choose appropriately
[Fff,Xx]=hist(VX_ice_eastward,numBins); %use hist function and get unnormalized values
PDF_DATA=Fff/trapz(Xx,Fff);
% figure; semilogy(Xx,PDF_DATA,'ko');%plot normalized histogram from the generated data

[xData, yData] = prepareCurveData( Xx, PDF_DATA );

%%% Fit Gaussian distribution 
ft = fittype( 'gauss1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0];
opts.StartPoint = [0.0368473769446665 5.78653806794038 9.4940247731043];

% Fit model to data.
[fitresult, ~] = fit( xData, yData, ft, opts );


Gaussian_coeff=coeffvalues(fitresult);
AA1=Gaussian_coeff(1);
BB1=Gaussian_coeff(2);
CC1=Gaussian_coeff(3);

% Plot fit with data.
figure( 'Name', 'Gaussian' );
h = plot( fitresult, xData, yData );
legend( h, 'PDF_DATA vs. Xx', 'Gaussian', 'Location', 'NorthEast' );
% Label axes
title ('Velocities Fluctuations','interpreter','latex','FontSize',12)
xlabel('Cross flow velocity fluctuations [cm/s]')
ylabel('PDF')
grid off
cd(strcat(HOME,'/output/single_particle/plots_NSIDC'))
filename = ['pdf_along',num2str(DAY_DISP*2+1),'.tif'];
print('-dtiff','-r0', filename)
hold off
close;
% 
% VVX=(find(VX_ice_eastward));
% VVX(isnan(VVX))=[];
% Moment1_mean = mean(VVX);
% Moment2_var = var(VVX);
% Moment3_skew = skewness(VVX);
% Moment4_kurt = kurtosis(VVX);
% fprintf('The distribution has a mean value of %d.\n',Moment1_mean);
% fprintf('The distribution has a variance velue of %d.\n',Moment2_var);
% fprintf('The distribution has a skewness value of %d.\n',Moment3_skew);
% fprintf('The distribution has a kurtosis value of %d.\n',Moment4_kurt);



%% PDF of vel fluctuations ACROSS
VY_ice_northward(isnan(VX_ice_eastward))=NaN;
histogram(VY_ice_northward,'Normalization','pdf'); %plot estimated pdf from the generated data

numBins=50; %choose appropriately
[Fff,Xx]=hist(VY_ice_northward,numBins); %use hist function and get unnormalized values
PDF_DATA=Fff/trapz(Xx,Fff);
% figure; semilogy(Xx,PDF_DATA,'ko');%plot normalized histogram from the generated data

[xData, yData] = prepareCurveData( Xx, PDF_DATA );

%%% Fit Gaussian distribution 
ft = fittype( 'gauss1' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
opts.Lower = [-Inf -Inf 0];
opts.StartPoint = [0.0685758279467839 -0.356975565035853 6.83045302566911];

% Fit model to data.
[fitresult, ~] = fit( xData, yData, ft, opts );


Gaussian_coeff=coeffvalues(fitresult);
AA1=Gaussian_coeff(1);
BB1=Gaussian_coeff(2);
CC1=Gaussian_coeff(3);

% Plot fit with data.
figure( 'Name', 'Gaussian' );
h = plot( fitresult, xData, yData );
legend( h, 'PDF_DATA vs. Xx', 'Gaussian', 'Location', 'NorthEast' );
% Label axes
title ('Fluctuation Velocities','interpreter','latex','FontSize',12)
xlabel('Cross flow velocity fluctuations [cm/s]')
ylabel('PDF')
grid off
cd(strcat(HOME,'/output/single_particle/plots_NSIDC'))
filename = ['pdf_across',num2str(DAY_DISP*2+1),'.tif'];
print('-dtiff','-r0', filename)
hold off
close;
close all;
% 
% VVY=(find(VY_ice_eastward));
% VVY(isnan(VVY))=[];
% Moment1_mean = mean(VVY);
% Moment2_var = var(VVY);
% Moment3_skew = skewness(VVY);
% Moment4_kurt = kurtosis(VVY);
% fprintf('The distribution has a mean value of %d.\n',Moment1_mean);
% fprintf('The distribution has a variance velue of %d.\n',Moment2_var);
% fprintf('The distribution has a skewness value of %d.\n',Moment3_skew);
% fprintf('The distribution has a kurtosis value of %d.\n',Moment4_kurt);





% % % % 
% % % % 
% % % % 
% % % % %%%
% % % % % visualize instantaneous fields
% % % % for ii=1:2%size(eastward_ice,3)
% % % % if any(any(eastward_ice(:,:,ii)))
% % % % %visualize data
% % % % cd('/Users/rosalinda/Documents/UCR/Research/ice_tracker/branch1/ArcticMappingTools');
% % % % %quiverpsn(lat,lon,zonal_ice(:,:,1),meridional_ice(:,:,1))
% % % % quiver(x_ice,y_ice,eastward_ice(:,:,ii),northward_ice(:,:,ii));
% % % % hold on; 
% % % % greenland
% % % % ylim([-2800000 0])
% % % % xlim([0 1300000])
% % % % hold on; 
% % % % 
% % % % % % print images
% % % % % cd(strcat(HOME,'/output/inst_field_NSIDC'))
% % % % % filename = [num2str(ii),'.tif'];
% % % % % print('-dtiff','-r0', filename)
% % % % % close;
% % % % end
% % % % end
% % % % 
% % % % scale=10000;
% % % % 
% % % % %mean vector 
% % % % quiver(X_trans(2),Y_trans(2),mean_ice_eastward_interp(2)*scale,mean_ice_northward_interp(2)*scale,'Color','red','LineWidth',2,'AutoScale','off')             
% % % % hold on;
% % % %    
% % % % %instantaneous vector
% % % % quiver(X_trans(2),Y_trans(2),east_ice(2)*scale,north_ice(2)*scale,'Color','black','LineWidth',2,'AutoScale','off')
% % % % 
% % % % %components
% % % % [THETA_I,R] = cart2pol(mean_ice_eastward_interp(2)*scale,mean_ice_northward_interp(2)*scale); %Convert to polar coordinates
% % % % 
% % % % %along
% % % % [xr,yr]=pol2cart(-THETA_I,VX_ice_eastward(2)*scale);
% % % % plot([X_trans(2) X_trans(2)+xr],[Y_trans(2) Y_trans(2)-yr],'--','Color',[.6 .1 1],'LineWidth',2); 
% % % % 
% % % % %across
% % % % THETA_II=-THETA_I+(pi/2);
% % % % [xxr,yyr]=pol2cart(-THETA_II,VY_ice_northward(2)*scale);
% % % % plot([X_trans(2)+xr X_trans(2)+xr-xxr],[Y_trans(2)-yr Y_trans(2)-yr-yyr],'--','Color',[1 0 .498],'LineWidth',2); 
% % % % 


% %check how close water and wind vectors are to each other
% for i=1:size(VY,1)* size(VX,2)
% if VY(i)~=0 
% 
% hold on;
% quiver(X_trans(i),Y_trans(i),VX(i)*100,VY(i)*100)
% hold on;
% quiver(water_x(i),water_y(i),east_water(i)*100,north_water(i)*100)
% hold off;
% pause
% 
% end
% end



cd(strcat(HOME,'/output/tracked_single'))
save(strcat('mean_map_u',num2str(DAY_DISP*2+1)),'mean_map_u');
save(strcat('mean_map_v',num2str(DAY_DISP*2+1)),'mean_map_v');
save(strcat('DX_ice_eastward_',num2str(DAY_DISP*2+1)),'DX_ice_eastward');
save(strcat('DY_ice_northward_',num2str(DAY_DISP*2+1)),'DY_ice_northward');
save(strcat('VX_ice_eastward_',num2str(DAY_DISP*2+1)),'VX_ice_eastward');
save(strcat('VY_ice_northward_',num2str(DAY_DISP*2+1)),'VY_ice_northward');
save(strcat('abs_disp_across_',num2str(DAY_DISP*2+1)),'abs_disp_across');
save(strcat('abs_disp_along_',num2str(DAY_DISP*2+1)),'abs_disp_along');
save(strcat('N_error_',num2str(DAY_DISP*2+1)),'N_error');
save(strcat('ACF2_u_',num2str(DAY_DISP*2+1)),'ACF2_u');
save(strcat('ACF2_v_',num2str(DAY_DISP*2+1)),'ACF2_v');
save(strcat('mean_ice_eastward_interp',num2str(DAY_DISP*2+1)),'mean_ice_eastward_interp');
save(strcat('mean_ice_northward_interp',num2str(DAY_DISP*2+1)),'mean_ice_northward_interp');
save(strcat('AA_',num2str(DAY_DISP*2+1)),'AA');
save(strcat('BB_',num2str(DAY_DISP*2+1)),'BB');
save(strcat('CC_',num2str(DAY_DISP*2+1)),'CC');
save(strcat('X_trans_',num2str(DAY_DISP*2+1)),'X_trans');
save(strcat('Y_trans_',num2str(DAY_DISP*2+1)),'Y_trans');
save(strcat('delta_tt_',num2str(DAY_DISP*2+1)),'delta_tt');
end





disp('FINISHED DISPERSION')
disp('BEGINING ROTATION')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ROTATION
%% Measure individual rotations 
%creates images with ice floes that are tracked only to later plot their
%trajectories
clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
close all

%% obtain centroid data
cd(strcat(HOME,'/output/tracked'))
x1 = importdata('x3.mat');
y1 = importdata('y3.mat');
FLOE_LIBRARY=importdata('FLOE_LIBRARY.mat');
properties = importdata('prop.mat');  
order = importdata('order.mat'); 
sat_order = importdata('sat_order.mat'); 



aqua=find(sat_order==0);   %aqua
terra=find(sat_order==1);  %terra

x=nan(size(x1,1),size(aqua,1)*2);
y=nan(size(x1,1),size(aqua,1)*2);


x(1:size(x1,1),1:size(x1,2))=x1;
y(1:size(x1,1),1:size(y1,2))=y1;


cd(strcat(HOME,'/input/info'))
[~,aqua_pass]  = xlsread('sat_2010.xlsx','A1:A173');
[~,terra_pass]  = xlsread('sat_2010.xlsx','B1:B173');


for i=1:size(aqua_pass,1)-1
    str1=aqua_pass{i,1};
    str2=aqua_pass{i+1,1};
    t2 = datevec(str2,'mmmm dd, yyyy HH:MM:SS');
    t1 = datevec(str1,'mmmm dd, yyyy HH:MM:SS');
    delta_t_aqua(i,1) = etime(t2,t1)/60;
end 

for i=1:size(terra_pass,1)-1
    str1=terra_pass{i,1};
    str2=terra_pass{i+1,1};
    t2 = datevec(str2,'mmmm dd, yyyy HH:MM:SS');
    t1 = datevec(str1,'mmmm dd, yyyy HH:MM:SS');
    delta_t_terra(i,1) = etime(t2,t1)/60;
end 


x_aqua=x(:,aqua);
y_aqua=y(:,aqua);
x_terra=x(:,terra);
y_terra=y(:,terra);

THETA_aqua=nan(size(x_aqua));
zeta_aqua=nan(size(x_aqua)); 
zeta_norm_aqua=nan(size(x_aqua));
SIZE_aqua=nan(size(x_aqua));
THETA_terra=nan(size(x_terra));
zeta_terra=nan(size(x_terra)); 
zeta_norm_terra=nan(size(x_terra));
SIZE_terra=nan(size(x_terra));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% aqua
for i = 1:size(x_aqua,1)
if any(x_aqua(i,:))
day=find(x_aqua(i,:));
[~,spaces]=size(day);
for f= 1:spaces-1
if (day(f+1)-day(f))-1==0   %0 spaces
delta_time=delta_t_aqua(day(f));
prop1=properties{1,aqua(day(f))};
prop2=properties{1,aqua(day(f+1))};
%find floe inside of props matrix
[row1,col1]=find(x_aqua(i,day(f))==prop1(:,6));
[row2,col2]=find(x_aqua(i,day(f+1))==prop2(:,6));

if isempty(row1)
N=x_aqua(i,day(f));
V=prop1(:,6);
[index a] = min(abs(N-V));

NN=y_aqua(i,day(f));
VV=prop1(:,7);
[index aa] = min(abs(NN-VV));

    if a==aa
        row1=a;
        x(i,aqua(day(f)))=prop1(aa,6);
        y(i,aqua(day(f)))=prop1(aa,7);
    end
end    
if isempty(row2)    
O=x_aqua(i,day(f+1));
W=prop2(:,6);
[index b] = min(abs(O-W));

OO=y_aqua(i,day(f+1));
WW=prop2(:,7);
[index bb] = min(abs(O-W));

    if b==bb
        row2=b;
        x(i,aqua(day(f+1)))=prop2(bb,6);
        y(i,aqua(day(f+1)))=prop2(bb,7);
    end
end

%min_value=min_scale*.25
%max_value=max_scale*.25
%prop1(row1,1)>min_value && prop2(row2,2)<min_value && prop1(row1,2)<max_value && prop2(row2,1)<max_value
if ~isempty(row1) && ~isempty(row2) 
floe1=[];
floe1=FLOE_LIBRARY{row1,aqua(day(f))};
   
floe2=[];
floe2=FLOE_LIBRARY{row2,aqua(day(f+1))};
    
floe1=uint8(255 *floe1);
floe2=uint8(255 *floe2);
im = double(floe1);      %original
imRot = double(floe2);   %rotated

if size(imRot,1)<16 || size(im,1)<16 || size(imRot,2)<16 || size(im,2)<16
    break
    
else
tformEstimate = imregcorr(im,imRot);
 
% if tformEstimate.T(1,1)==1 && tformEstimate.T(1,2)==0
% THETA_aqua(i,day(f))=0;
% zeta_aqua(i,day(f))=0;
% zeta_norm_aqua(i,day(f))=0;

% else
% setup intensity-based image registration
[optimizer,metric] = imregconfig('monomodal');

% register imRot with im so that cameraman is upright
imReg = imregister(imRot, im, 'rigid', optimizer, metric);
trans=imregtform(imRot, im, 'rigid', optimizer, metric);
transformation=trans.T;

theta1=acosd(transformation(1,1));
theta2=asind(transformation(1,2));
   
% % figure,imshowpair(im,imRot,'montage');
% % figure; imshowpair(im,imReg,'montage');


%test result
im_new=imrotate(imRot,-theta2);
tformEstimate = imregcorr(im,im_new);

% setup intensity-based image registration
[optimizer,metric] = imregconfig('monomodal');

%only rotation
% register imRot with im so that cameraman is upright
im_newReg = imregister(im_new, im, 'Translation', optimizer, metric);
trans=imregtform(im_new, im, 'rigid', optimizer, metric);
transformation=trans.T;

theta11=acosd(transformation(1,1));
theta22=asind(transformation(1,2));

% figure,imshowpair(im,im_new,'montage');
% figure; imshowpair(im,im_newReg,'montage');

compose = imfuse(im,im_newReg,'falsecolor','Scaling','joint','ColorChannels',[1 2 1]);
%figure,imshow(compose)
   
comp1=find(compose(:,:,1)~=0);
comp2=find(compose(:,:,2)~=0);
comp3=find(compose(:,:,3)~=0);
white=intersect(intersect(comp1,comp2),comp3);
   
blank1=zeros(size(compose));
blank1(white)=1;
%figure,imshow(blank1)
   
comp11=comp1;
comp22=comp2;
comp33=comp3;
   
comp11(ismember(comp11,white))=[];
comp22(ismember(comp22,white))=[];
comp33(ismember(comp33,white))=[];
   
blank2=zeros(size(compose));
blank2(comp11)=1;
blank2(comp22)=1;
blank2(comp33)=1;
% % figure,imshow(blank2)
      
match1=blank1(:,:,1);
no_match1=blank2(:,:,1);
   
match_A=length(find(match1==1));
no_match_A=length(find(no_match1==1));
   
   if abs(no_match_A/match_A)>.2 ...
      || (abs(theta2)>60 && abs(no_match_A/match_A)>.15) 
   area_match=[];
      
    parfor (z=50:310,M_WORKERS)
    new=imrotate(imRot,z);
      
    tformEstimate = imregcorr(im,new);
 
    % setup intensity-based image registration
    [optimizer,metric] = imregconfig('monomodal');

    % only translation
    % register imRot with im so that cameraman is upright
    im_newReg = imregister(new, im, 'Translation', optimizer, metric);
           
    compose = imfuse(im,im_newReg,'falsecolor','Scaling','joint','ColorChannels',[1 2 1]);
    %figure,imshow(compose)
   
    comp1=find(compose(:,:,1)~=0);
    comp2=find(compose(:,:,2)~=0);
    comp3=find(compose(:,:,3)~=0);
    white=intersect(intersect(comp1,comp2),comp3);
   
    blank1=zeros(size(compose));
    blank1(white)=1;
    %figure,imshow(blank1)
   
    comp11=comp1;
    comp22=comp2;
    comp33=comp3;
   
    comp11(ismember(comp11,white))=[];
    comp22(ismember(comp22,white))=[];
    comp33(ismember(comp33,white))=[];
   
    blank2=zeros(size(compose));
    blank2(comp11)=1;
    blank2(comp22)=1;
    blank2(comp33)=1;
    %figure,imshow(blank2)
     
    match1=blank1(:,:,1);
    no_match1=blank2(:,:,1);
   
    match=length(find(match1==1));
    no_match=length(find(no_match1==1));
       
    area_match(z)=abs(no_match/match); 
    end
  
    %compare to no rotation
    z=1;
    new=imrotate(imRot,0);
      
    tformEstimate = imregcorr(im,new);
 
    % setup intensity-based image registration
    [optimizer,metric] = imregconfig('monomodal');

    % only translation
    % register imRot with im so that cameraman is upright
    im_newReg = imregister(new, im, 'Translation', optimizer, metric);
           
    compose = imfuse(im,im_newReg,'falsecolor','Scaling','joint','ColorChannels',[1 2 1]);
    %figure,imshow(compose)
   
    comp1=find(compose(:,:,1)~=0);
    comp2=find(compose(:,:,2)~=0);
    comp3=find(compose(:,:,3)~=0);
    white=intersect(intersect(comp1,comp2),comp3);
   
    blank1=zeros(size(compose));
    blank1(white)=1;
    %figure,imshow(blank1)
   
    comp11=comp1;
    comp22=comp2;
    comp33=comp3;
   
    comp11(ismember(comp11,white))=[];
    comp22(ismember(comp22,white))=[];
    comp33(ismember(comp33,white))=[];
   
    blank2=zeros(size(compose));
    blank2(comp11)=1;
    blank2(comp22)=1;
    blank2(comp33)=1;
    %figure,imshow(blank2)
     
    match1=blank1(:,:,1);
    no_match1=blank2(:,:,1);
   
    match=length(find(match1==1));
    no_match=length(find(no_match1==1));
       
    area_match(z)=abs(no_match/match); 
    revised=[];  
     
      if min(area_match(50:310))<.16 ...
      && min(area_match(50:310))<area_match(1)
      revised=min(49+find(area_match(50:310)==min(area_match(50:310))));
         if revised>180 ...
         theta_revised=-(360-revised);
         final_theta=-theta_revised;
         else
         theta_revised=revised;
         final_theta=-theta_revised;
         end 
      elseif area_match(1)<.1
      theta_revised=0;
      final_theta=0;
      elseif min(area_match(50:310))>abs(no_match_A/match_A)...
      && abs(no_match_A/match_A)<.22
      theta_revised=-theta2;
      final_theta=theta2;
      else
      theta_revised=NaN;    
      final_theta=NaN;
      end
      
     if abs(theta_revised)>90  && abs(theta_revised)<270 
          if abs(((no_match_A/match_A)-area_match(revised))/(no_match_A/match_A))<.33 ...
             && no_match_A/match_A <.25
             theta_revised=-theta2;
             final_theta=theta2;
          else
             theta_revised=NaN;
             final_theta=NaN;  
          end
      end
            
   Rotation_angle=theta_revised;

   else
   Rotation_angle=-theta2;
   final_theta=theta2;
   end
   
   THETA_aqua(i,day(f))=final_theta;
   zeta_aqua(i,day(f))=(final_theta/delta_time);
   
   zeta_norm_aqua(i,day(f))=zeta_aqua(i,day(f))*(pi/(60*180*(1.3*10^-4)));

   if isnan(final_theta) 
   SIZE_aqua(i,day(f))=NaN;
   else
   SIZE_aqua(i,day(f))=sqrt(prop2(row2,1))*.250;  %km
   end
   

end
end
end
end
end
end

cd(strcat(HOME,'/output/tracked'))
save(['THETA_aqua_before'],'THETA_aqua');
save(['zeta_aqua_before'],'zeta_aqua');
save(['zeta_norm_aqua_before'],'zeta_norm_aqua');
save(['SIZE_aqua_before'],'SIZE_aqua'); 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% terra

cent_x=[];
cent_y=[];
[d,~]=size(x_terra);
prop1=[];
prop2=[];

for i = 1:size(x_terra,1)
if any(x_terra(i,:))
day=find(x_terra(i,:));
[~,spaces]=size(day);
for f= 1:spaces-1
if (day(f+1)-day(f))-1==0   %0 spaces
delta_time=delta_t_terra(day(f));
prop1=properties{1,terra(day(f))};
prop2=properties{1,terra(day(f+1))};
%find floe inside of props matrix
[row1,col1]=find(x_terra(i,day(f))==prop1(:,6));
[row2,col2]=find(x_terra(i,day(f+1))==prop2(:,6));



if isempty(row1)
N=x_terra(i,day(f));
V=prop1(:,6);
[index a] = min(abs(N-V));

NN=y_terra(i,day(f));
VV=prop1(:,7);
[index aa] = min(abs(NN-VV));

    if a==aa
        row1=a;
        x(i,terra(day(f)))=prop1(aa,6);
        y(i,terra(day(f)))=prop1(aa,7);
    end
end    
if isempty(row2)    
O=x_terra(i,day(f+1));
W=prop2(:,6);
[index b] = min(abs(O-W));

OO=y_terra(i,day(f+1));
WW=prop2(:,7);
[index bb] = min(abs(O-W));

    if b==bb
        row2=b;
        x(i,terra(day(f+1)))=prop2(bb,6);
        y(i,terra(day(f+1)))=prop2(bb,7);
    end
end


%min_value=min_scale*.25
%max_value=max_scale*.25
%prop1(row1,1)>min_value && prop2(row2,2)<min_value && prop1(row1,2)<max_value && prop2(row2,1)<max_value
if ~isempty(row1) && ~isempty(row2) 
floe1=[];
floe1=FLOE_LIBRARY{row1,terra(day(f))};
           
floe2=[];
floe2=FLOE_LIBRARY{row2,terra(day(f+1))};
        
floe1=uint8(255 *floe1);
floe2=uint8(255 *floe2);
im = double(floe1);      %original
imRot = double(floe2);   %rotated

if size(imRot,1)<16 || size(im,1)<16 || size(imRot,2)<16 || size(im,2)<16
    break
    
else

tformEstimate = imregcorr(im,imRot);

% if tformEstimate.T(1,1)==1 && tformEstimate.T(1,2)==0
% THETA_terra(i,day(f))=0;
% zeta_terra(i,day(f))=0;
% zeta_norm_terra(i,day(f))=0;
% 
% else
% setup intensity-based image registration
[optimizer,metric] = imregconfig('monomodal');

% register imRot with im so that cameraman is upright
imReg = imregister(imRot, im, 'rigid', optimizer, metric);
trans=imregtform(imRot, im, 'rigid', optimizer, metric);
transformation=trans.T;

theta1=acosd(transformation(1,1));
theta2=asind(transformation(1,2));
   
% % figure,imshowpair(im,imRot,'montage');
% % figure; imshowpair(im,imReg,'montage');


%test result
im_new=imrotate(imRot,-theta2);
tformEstimate = imregcorr(im,im_new);

% setup intensity-based image registration
[optimizer,metric] = imregconfig('monomodal');

%only rotation
% register imRot with im so that cameraman is upright
im_newReg = imregister(im_new, im, 'Translation', optimizer, metric);
trans=imregtform(im_new, im, 'rigid', optimizer, metric);
transformation=trans.T;

theta11=acosd(transformation(1,1));
theta22=asind(transformation(1,2));

% figure,imshowpair(im,im_new,'montage');
% figure; imshowpair(im,im_newReg,'montage');

compose = imfuse(im,im_newReg,'falsecolor','Scaling','joint','ColorChannels',[1 2 1]);
%figure,imshow(compose)
   
comp1=find(compose(:,:,1)~=0);
comp2=find(compose(:,:,2)~=0);
comp3=find(compose(:,:,3)~=0);
white=intersect(intersect(comp1,comp2),comp3);
   
blank1=zeros(size(compose));
blank1(white)=1;
%figure,imshow(blank1)
   
comp11=comp1;
comp22=comp2;
comp33=comp3;
   
comp11(ismember(comp11,white))=[];
comp22(ismember(comp22,white))=[];
comp33(ismember(comp33,white))=[];
   
blank2=zeros(size(compose));
blank2(comp11)=1;
blank2(comp22)=1;
blank2(comp33)=1;
% % figure,imshow(blank2)
      
match1=blank1(:,:,1);
no_match1=blank2(:,:,1);
   
match_A=length(find(match1==1));
no_match_A=length(find(no_match1==1));
   
   if abs(no_match_A/match_A)>.2 ...
      || (abs(theta2)>60 && abs(no_match_A/match_A)>.15) 
   area_match=[];
      
    parfor (z=50:310,M_WORKERS)
    new=imrotate(imRot,z);
      
    tformEstimate = imregcorr(im,new);
 
    % setup intensity-based image registration
    [optimizer,metric] = imregconfig('monomodal');

    % only translation
    % register imRot with im so that cameraman is upright
    im_newReg = imregister(new, im, 'Translation', optimizer, metric);
           
    compose = imfuse(im,im_newReg,'falsecolor','Scaling','joint','ColorChannels',[1 2 1]);
    %figure,imshow(compose)
   
    comp1=find(compose(:,:,1)~=0);
    comp2=find(compose(:,:,2)~=0);
    comp3=find(compose(:,:,3)~=0);
    white=intersect(intersect(comp1,comp2),comp3);
   
    blank1=zeros(size(compose));
    blank1(white)=1;
    %figure,imshow(blank1)
   
    comp11=comp1;
    comp22=comp2;
    comp33=comp3;
   
    comp11(ismember(comp11,white))=[];
    comp22(ismember(comp22,white))=[];
    comp33(ismember(comp33,white))=[];
   
    blank2=zeros(size(compose));
    blank2(comp11)=1;
    blank2(comp22)=1;
    blank2(comp33)=1;
    %figure,imshow(blank2)
     
    match1=blank1(:,:,1);
    no_match1=blank2(:,:,1);
   
    match=length(find(match1==1));
    no_match=length(find(no_match1==1));
       
    area_match(z)=abs(no_match/match); 
    end
  
    %compare to no rotation
    z=1;
    new=imrotate(imRot,0);
      
    tformEstimate = imregcorr(im,new);
 
    % setup intensity-based image registration
    [optimizer,metric] = imregconfig('monomodal');

    % only translation
    % register imRot with im so that cameraman is upright
    im_newReg = imregister(new, im, 'Translation', optimizer, metric);
           
    compose = imfuse(im,im_newReg,'falsecolor','Scaling','joint','ColorChannels',[1 2 1]);
    %figure,imshow(compose)
   
    comp1=find(compose(:,:,1)~=0);
    comp2=find(compose(:,:,2)~=0);
    comp3=find(compose(:,:,3)~=0);
    white=intersect(intersect(comp1,comp2),comp3);
   
    blank1=zeros(size(compose));
    blank1(white)=1;
    %figure,imshow(blank1)
   
    comp11=comp1;
    comp22=comp2;
    comp33=comp3;
   
    comp11(ismember(comp11,white))=[];
    comp22(ismember(comp22,white))=[];
    comp33(ismember(comp33,white))=[];
   
    blank2=zeros(size(compose));
    blank2(comp11)=1;
    blank2(comp22)=1;
    blank2(comp33)=1;
    %figure,imshow(blank2)
     
    match1=blank1(:,:,1);
    no_match1=blank2(:,:,1);
   
    match=length(find(match1==1));
    no_match=length(find(no_match1==1));
       
    area_match(z)=abs(no_match/match); 
    revised=[];  
     
      if min(area_match(50:310))<.16 ...
      && min(area_match(50:310))<area_match(1)
      revised=min(49+find(area_match(50:310)==min(area_match(50:310))));
         if revised>180 ...
         theta_revised=-(360-revised);
         final_theta=-theta_revised;
         else
         theta_revised=revised;
         final_theta=-theta_revised;
         end 
      elseif area_match(1)<.1
      theta_revised=0;
      final_theta=0;
      elseif min(area_match(50:310))>abs(no_match_A/match_A)...
      && abs(no_match_A/match_A)<.22
      theta_revised=-theta2;
      final_theta=theta2;
      else
      theta_revised=NaN;    
      final_theta=NaN;
      end
      
      if abs(theta_revised)>90  && abs(theta_revised)<270 
          if abs(((no_match_A/match_A)-area_match(revised))/(no_match_A/match_A))<.33 ...
             && no_match_A/match_A <.25
             theta_revised=-theta2;
             final_theta=theta2;
          else
             theta_revised=NaN;
             final_theta=NaN;  
          end
      end
            
   Rotation_angle=theta_revised;

   else
   Rotation_angle=-theta2;
   final_theta=theta2;
   end
   
   THETA_terra(i,day(f))=final_theta;
   zeta_terra(i,day(f))=(final_theta/delta_time);
   zeta_norm_terra(i,day(f))=zeta_terra(i,day(f))*(pi/(60*180*(1.3*10^-4)));

   if isnan(final_theta) 
   SIZE_terra(i,day(f))=NaN;
   else
   SIZE_terra(i,day(f))=sqrt(prop2(row2,1))*.250;  %km
   end
   
end
end
end
end
end
end

cd(strcat(HOME,'/output/tracked'))
save('THETA_terra_before','THETA_terra');
save('zeta_terra_before','zeta_terra');
save('zeta_norm_terra_before','zeta_norm_terra');
save('SIZE_terra_before','SIZE_terra'); 





%% obtain centroid data
cd(strcat(HOME,'/output/tracked'))
THETA_aqua = importdata('THETA_aqua_before.mat');
zeta_aqua = importdata('zeta_aqua_before.mat');
zeta_norm_aqua = importdata('zeta_norm_aqua_before.mat');
SIZE_aqua = importdata('SIZE_aqua_before.mat');

zeta_norm_aqua(find(zeta_norm_aqua==0))=NaN;
zeta_norm_pdf1_aqua=zeta_norm_aqua(find(~isnan(zeta_norm_aqua)));
SIZE_aqua(find(isnan(zeta_norm_aqua)))=NaN;
SIZE_aqua1=SIZE_aqua(find(~isnan(SIZE_aqua)));

THETA_terra = importdata('THETA_terra_before.mat');
zeta_terra = importdata('zeta_terra_before.mat');
zeta_norm_terra = importdata('zeta_norm_terra_before.mat');
SIZE_terra = importdata('SIZE_terra_before.mat');

zeta_norm_terra(find(zeta_norm_terra==0))=NaN;
zeta_norm_pdf1_terra=zeta_norm_terra(find(~isnan(zeta_norm_terra)));
SIZE_terra(find(isnan(zeta_norm_terra)))=NaN;
SIZE_terra1=SIZE_terra(find(~isnan(SIZE_terra)));

order = importdata('order.mat'); 
sat_order = importdata('sat_order.mat'); 
x= importdata('x3.mat');
y= importdata('y3.mat');

aqua=find(sat_order==0);   %aqua
terra=find(sat_order==1);  %terra


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%condition to only account for non-interactive ice floes
x_aqua=x(:,aqua);
y_aqua=y(:,aqua);
x_terra=x(:,terra);
y_terra=y(:,terra);


cd(strcat(HOME,'/output/tracked'))
%%% Plot for terra
%Plot PDF's of floes' vorticities
hist1=histogram(zeta_norm_pdf1_aqua)
%,'Normalization','pdf');
%a=hist1.NumBins;
%hist1(1).FaceColor = [.8 .8 1];

%[f,xx]=hist(zeta_norm_pdf1_aqua,a);
%bar(xx,-f/trapz(xx))
title ('Hist: Relative vorticity/\it f','interpreter','latex','FontSize',16)
xlabel('$\zeta$ / \it f','interpreter','latex','FontSize',14);
ylabel ('Number count','interpreter','latex','FontSize',14)
filename = 'rel_dispersion_aqua.tif';
print('-dtiff','-r250',filename)
hold off
close;

%Plot scatterplot's of floes' vorticities
figure,scatter(SIZE_aqua1,zeta_norm_pdf1_aqua)
title ('Length scale vs Relative vorticity/\it f','interpreter','latex','FontSize',16)
xlabel ('Length scale [km]','interpreter','latex','FontSize',14)
ylabel('$\zeta$ / \it f','interpreter','latex','FontSize',14);
filename = 'scatter_rot_aqua.tif';
print('-dtiff','-r250',filename)
hold off
close;

%% variance
bins=linspace(floor(min(SIZE_aqua1)),ceil(max(SIZE_aqua1)),12);
for i=1:size(bins,2)-1
    A=find(SIZE_aqua1>bins(i));
    B=find(SIZE_aqua1<=bins(i+1));
    C=intersect(A,B);
    numb_perbin_aqua(i)=numel(zeta_norm_pdf1_aqua(C));
    values_perbin_aqua(1:numb_perbin_aqua(i),i)=zeta_norm_pdf1_aqua(C);

    if numb_perbin_aqua(i)<5
    var_aqua(i)=NaN;   
    else
    var_aqua(i)=var(zeta_norm_pdf1_aqua(C))/1.3*10^-4;
    end
    bins_size_aqua(i)=1/bins(i);
end

figure,scatter(bins_size_aqua,var_aqua)
title ('Aqua-Length scale vs Variance normalized by $f^{2}$','interpreter','latex','FontSize',16)
xlabel ('Length scale$^{-1}$ [km$^{-1}]$','interpreter','latex','FontSize',14)
ylabel('Variance normalized by $f^{2}$','interpreter','latex','FontSize',14);
set(gca,'xscale','log', 'yscale','log')
xlim([1e-2,1e0])
filename = ('scatter_variance_aqua.tif');
print('-dtiff','-r250',filename)
hold off
close;


%%% Plot for terra
%Plot PDF's of floes' vorticities
hist1=histogram(zeta_norm_pdf1_terra);
%,'Normalization','pdf');
%a=hist1.NumBins;
%hist1(1).FaceColor = [.8 .8 1];

%[f,xx]=hist(zeta_norm_pdf1_aqua,a);
%bar(xx,-f/trapz(xx))
title ('Hist: Relative vorticity/\it f','interpreter','latex','FontSize',16)
xlabel('$\zeta$ / \it f','interpreter','latex','FontSize',14);
ylabel ('Number count','interpreter','latex','FontSize',14)
filename = 'rel_dispersion_terra.tif';
print('-dtiff','-r250',filename)
hold off
close;

%Plot scatterplot's of floes' vorticities
figure,scatter(SIZE_terra1,zeta_norm_pdf1_terra)
title ('Length scale vs Relative vorticity/\it f','interpreter','latex','FontSize',16)
xlabel ('Length scale [km]','interpreter','latex','FontSize',14)
ylabel('$\zeta$ / \it f','interpreter','latex','FontSize',14);
filename = 'scatter_rot_terra.tif';
print('-dtiff','-r250',filename)
hold off
close;

clear bins var_terra_I
%% variance
bins=linspace(floor(min(SIZE_terra1)),ceil(max(SIZE_terra1)),9);
for i=1:size(bins,2)-1
    A=find(SIZE_terra1>bins(i));
    B=find(SIZE_terra1<bins(i+1));
    C=intersect(A,B);
    numb_perbin_terra(i)=numel(zeta_norm_pdf1_terra(C));
    values_perbin_terra(1:numb_perbin_terra(i),i)=zeta_norm_pdf1_terra(C);
    if numb_perbin_terra(i)<5
    var_terra(i)=NaN;   
    else
    var_terra(i)=var(zeta_norm_pdf1_terra(C))/1.3*10^-4;
    end    
    bins_size_terra(i)=1/bins(i);
end

figure,scatter(bins_size_terra,var_terra)
title ('Terra-Length scale vs Variance normalized by $f^{2}$','interpreter','latex','FontSize',16)
xlabel ('Length scale$^{-1}$ [km$^{-1}]$','interpreter','latex','FontSize',14)
ylabel('Variance normalized by $f^{2}$','interpreter','latex','FontSize',14);
set(gca,'xscale','log', 'yscale','log')
xlim([1e-2,1e0])
filename = 'scatter_variance_terra.tif';
print('-dtiff','-r250',filename)
hold off
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot rotation angle measured
close all
THETA=zeros(size(x));

for i=1:size(THETA_aqua,2)
THETA(1:size(THETA_aqua(:,i),1),aqua(i))=THETA_aqua(:,i);
end 

for i=1:size(THETA_terra,2)
THETA(1:size(THETA_terra(:,i),1),terra(i))=THETA_terra(:,i);
end 


ZETA_norm=zeros(size(x));

for i=1:size(zeta_norm_aqua,2)
ZETA_norm(1:size(zeta_norm_aqua(:,i),1),aqua(i))=zeta_norm_aqua(:,i);
end 

for i=1:size(zeta_norm_terra,2)
ZETA_norm(1:size(zeta_norm_terra(:,i),1),terra(i))=zeta_norm_terra(:,i);
end 


%%% on raw images
cd(strcat(HOME,'/output/contours'))
im_files = dir('*.tif');
INDEX=0;


%title date
tic
for i=1:size(im_files,1)/2
m{1,i}=datedisp(BEGINNING+i);
end
toc
m = repelem(m,2);

for i = order
INDEX=INDEX+1;    
cd(strcat(HOME,'/output/contours'))
fnm = im_files(i).name; %call image to plot on
im = imread(fnm);

TITLE={};
if contains(fnm,'aqua')
TITLE= strcat(m(INDEX),':  Aqua');
elseif contains(fnm,'terra')
TITLE= strcat(m(INDEX),':  Terra');
end

[n,o,~]= size(im);
figure, imshow(im); hold on
%set(gcf,'Visible','off');
for j=1:size(x,1)
    if x(j,INDEX)~=0 && THETA(j,INDEX)~=0
    text(x(j,INDEX),y(j,i), strcat(num2str(THETA(j,INDEX)),'\circ'), 'clipping', 'off','fontsize',14,'Color','red','FontName', 'Arial', 'FontWeight','bold');
    end
end


if contains(fnm,'aqua')
    cd(strcat(HOME,'/output/rotation_aqua'))
else
    cd(strcat(HOME,'/output/rotation_terra'))
end
set(gcf,'units','inches','position',[0,0,12,8])

filename = [num2str(INDEX),'_',fnm];
print('-dtiff','-r0', filename)
hold off
close;
end


%%% on BW images
cd(strcat(HOME,'/output/raw_black'))
im_files = dir('*.tif');
INDEX=0;

%title date
for i=1:size(im_files,1)/2
m{1,i}=datedisp([BEGINNING+i]);
end
m = repelem(m,2);

for i = order
INDEX=INDEX+1;    
cd(strcat(HOME,'/output/raw_black'))
fnm = im_files(i).name; %call image to plot on
im = imread(fnm);

TITLE={};
if contains(fnm,'aqua')
TITLE= strcat(m(INDEX),':  Aqua');
elseif contains(fnm,'terra')
TITLE= strcat(m(INDEX),':  Terra');
end

[n,o,~]= size(im);
figure, imshow(im); hold on
%set(gcf,'Visible','off');
for j=1:size(x,1)
    if x(j,INDEX)~=0 && THETA(j,INDEX)~=0
    text(x(j,INDEX),y(j,i), strcat(num2str(THETA(j,INDEX)),'\circ'), 'clipping', 'off','fontsize',14,'Color','red','FontName', 'Arial', 'FontWeight','bold');
    end
end

if contains(fnm,'aqua')
    cd(strcat(HOME,'/output/rotation1_aqua'))
else
    cd(strcat(HOME,'/output/rotation1_terra'))
end

filename = [num2str(INDEX),'_',fnm];
print('-dtiff','-r0', filename)
hold off;
close;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%make a combined database from each satellite
if size(THETA_terra,1)>= size(THETA_aqua,1)
   DATA_THETA=zeros([size(THETA_terra,1),size(THETA_terra,2)]);
   THETA_aqua1=DATA_THETA;
   THETA_aqua1(1:size(THETA_aqua,1),1:size(THETA_aqua,2))=THETA_aqua;
   THETA_terra1=THETA_terra;
elseif size(THETA_terra,1)< size(THETA_aqua,1)
   DATA_THETA=zeros([size(THETA_aqua,1),size(THETA_aqua,2)]);
   THETA_terra1=DATA_THETA;
   THETA_terra1(1:size(THETA_terra,1),1:size(THETA_terra,2))=THETA_terra;
   THETA_aqua1=THETA_aqua;
end


%put THETA_terra and THETA_aqua into a single variable with 7 deg to check
for i=1:size(THETA_terra1,1)
for j=1:size(THETA_terra1,2)
    if THETA_terra1(i,j)~=0 && ~isnan(THETA_terra1(i,j))...
     && THETA_aqua1(i,j)~=0 && ~isnan(THETA_aqua1(i,j))...
     && ( (THETA_aqua1(i,j)>0 && THETA_terra1(i,j)>0) ||...
          (THETA_aqua1(i,j)<0 && THETA_terra1(i,j)<0) )...
     && (THETA_aqua1(i,j)-THETA_terra1(i,j))<7
     DATA_THETA(i,j)= mean([THETA_aqua1(i,j),THETA_terra1(i,j)]);
    elseif THETA_terra1(i,j)==0 || isnan(THETA_terra1(i,j))...
           && THETA_aqua1(i,j)~=0 && ~isnan(THETA_aqua1(i,j))
     DATA_THETA(i,j)= THETA_aqua1(i,j);
    elseif THETA_aqua1(i,j)==0 || isnan(THETA_aqua1(i,j))...
           && THETA_terra1(i,j)~=0 && ~isnan(THETA_terra1(i,j))
     DATA_THETA(i,j)= THETA_terra1(i,j);

    end
end    
end

cd(strcat(HOME,'/output/tracked'))
save('DATA_THETA','DATA_THETA');







% %% Mean vel + bath
% clearvars -except HOME BEGINNING ROWS LIST1 LIST2 EXP1 EXP2 M_WORKERS LAND water_file wind_file nsidc_file
% close all
% 
% cd(strcat(HOME,'/output/BW_concatenated/'))
% im_files = dir('*.tif');
% 
% fnm1 = im_files(1).name; %call image to get icefloe image
% im = imread(fnm1);
% 
% 
% Y1_region=(1:size(im,1));
% X1_region=(1:size(im,2));
% 
%  
% [X_region,Y_region]=meshgrid(X1_region,Y1_region);
% X_region(:,:)=1;
% Y_region(:,:)=1;
% 
% 
% cd(strcat(HOME,'/output/tracked'))
% u_velocity= importdata('u_filt.mat');
% v_velocity= importdata('v_filt.mat');
% x=importdata('x_vel.mat');
% y=importdata('y_vel.mat');
% final_tracker= importdata('final_tracker.mat');
% 
% cd(strcat(HOME,'/output/tracked'))
% delta_t=importdata('delta_t.mat');
% 
% 
% [a,b]=size(X_region);
% compMean_u=[];
% 
% %allowed factors for h:3392,1696,848,424,64,53,32,16,8,4,2,1
% %allowed factors for 
% %g: 5680,2840,420,1136,710,368,355,284,142,80,71,40,20,16,10,8,5,4,2,1
% 
% g=80; %number of row in compartment 5680 height
% h=64; %number of column in compartment 3392 width
% 
% g1=a/g;
% h1=b/h;
% 
% %% average u
% 
% rowId = ceil( (1 : a) / g );
% colId = ceil( (1 : b) / h );
% [colID, rowID] = meshgrid( colId, rowId );
% 
% compartmentID = colID + (rowID - 1) * max(colId);
% 
% 
% % for i=1:compartmentID(a*b)
% % [A,B]=find(compartmentID==i);
% % comp_row{i}=A;
% % comp_col{i}=B;
% % end
% 
% 
% [d,e]=size(x);
% for i=1:d
%    for j=1:e
%    if x(i,j)~=0
%    [comp]=[y(i,j),x(i,j)];
%    comp=round(comp);
%    X_comp(i,j)=compartmentID(comp(1),comp(2));
%    end
%    end
% end
%        
% 
% COMP=cell(size(1:compartmentID(a*b)));
% 
% %account for 1 day-skip 
% %% velocity data [km/day]
% u=[];
% v=[];
% 
% [c,~] = size(x);
% 
% for i=1:c
%     if any(x(i,:))
%        day=find(x(i,:)); 
%        [~,spaces]=size(day);
%        for f= 1:spaces-1
%            if (day(f+1)-day(f))-1==3       %3 spaces
%               delta_time=delta_t(day(f)+3) + delta_t(day(f)+2) + ... 
%               delta_t(day(f)+1) + delta_t(day(f));
%               u(i,day(f)) = ((x(i,day(f+1))-x(i,day(f)))*0.250*60*24)/(delta_time);
%               v(i,day(f)) = ((y(i,day(f+1))-y(i,day(f)))*0.250*60*24)/(delta_time);
%               COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1)+1,1)= u(i,day(f)); 
%               COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1),2)= v(i,day(f)); 
%                                       
%            elseif (day(f+1)-day(f))-1==2   %2 spaces
%               delta_time=delta_t(day(f)+2) + delta_t(day(f)+1) + ... 
%               delta_t(day(f));
%               u(i,day(f)) = ((x(i,day(f+1))-x(i,day(f)))*0.250*60*24)/(delta_time);
%               v(i,day(f)) = ((y(i,day(f+1))-y(i,day(f)))*0.250*60*24)/(delta_time);
%               COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1)+1,1)= u(i,day(f)); 
%               COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1),2)= v(i,day(f)); 
%                    
%                    
%           elseif (day(f+1)-day(f))-1==1   %1 space
%              delta_time=delta_t(day(f)+1) + delta_t(day(f));
%              u(i,day(f)) = ((x(i,day(f+1))-x(i,day(f)))*0.250*60*24)/(delta_time);
%              v(i,day(f)) = ((y(i,day(f+1))-y(i,day(f)))*0.250*60*24)/(delta_time);
%              COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1)+1,1)= u(i,day(f)); 
%              COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1),2)= v(i,day(f)); 
%                                           
%           elseif (day(f+1)-day(f))-1==0   %0 spaces
%              delta_time=delta_t(day(f));
%              u(i,day(f)) = ((x(i,day(f+1))-x(i,day(f)))*0.250*60*24)/(delta_time);
%              v(i,day(f)) = ((y(i,day(f+1))-y(i,day(f)))*0.250*60*24)/(delta_time);    
%              COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1)+1,1)= u(i,day(f)); 
%              COMP{X_comp(i,day(f))}(size(COMP{X_comp(i,day(f))},1),2)= v(i,day(f)); 
%           end 
%        end
%     end
% end
%         
%        
% 
% %check comp to see how many velocities per bin are getting avergaed
% 
% %get average of each compartment
% for i=1:compartmentID(a*b)
%     if ~isempty(COMP{1,i})
%     mean_u(i)=mean(COMP{1,i}(:,1));
%     mean_v(i)=mean(COMP{1,i}(:,2));
%     else
%     mean_u(i)=0;
%     mean_v(i)=0;
%     end    
% end
% 
% index=1;
% x_center=h/2:h:b;
% y_center=g/2:g:a;
% 
% for i=1:size(y_center,2)
% for j=1:size(x_center,2)
%     X(index)=x_center(1,j);
%     Y(index)=y_center(1,i);
%     index=index+1;
% end
% end
% X=round(X);
% Y=round(Y);
% 
% Y1=-Y;
% mean_v1=-mean_v;             %negative denotes southward motion
% 
% mean_u(~isfinite(mean_u))=0;
% mean_v1(~isfinite(mean_v1))=0;
% 
% 
% % mean_u(find(mean_v1==0))=NaN;
% % mean_v1(find(mean_v1==0))=NaN;
% 
% 
% %% post-processing, meadian test + std + interpolation
% % look at 8 neighbors, if it passes the median test, this is ok
% % otherwise do bilinear interpolation 
% x=num2cell(X',1)
% y=num2cell(Y1',1);
% u=num2cell(mean_u',1);
% v=num2cell(mean_v1',1);
% 
% scatter(u{1,1},v{1,1})
% % print images
% cd(strcat(HOME,'/output/tracked'))
% print('-dtiff','-r250', 'outliers_mean.tif')
% 
% typevector=x; %typevector will be 1 for regular vectors, 0 for masked areas
% counter=0;
% 
% %% PIV postprocessing loop
% % Settings
% umin = -2*std(u{1,1}); % minimum allowed u velocity
% umax = 2*std(u{1,1}); % maximum allowed u velocity
% vmin = -2*std(v{1,1}); % minimum allowed v velocity
% vmax = 2*std(v{1,1}); % maximum allowed v velocity
% stdthresh=5; % threshold for standard deviation check
% epsilon=0.15; % epsilon for normalized median test
% thresh=3; % threshold for normalized median test
% 
% cd(strcat(HOME,'/input/info'))
% u_filt=cell(1,1);
% v_filt=u_filt;
% typevector_filt=u_filt;
% for PIVresult=1:size(x,1)
%     u_filtered=u{PIVresult,1};
%     v_filtered=v{PIVresult,1};
%     typevector_filtered=typevector{PIVresult,1};
%     %vellimit check
%     u_filtered(u_filtered<umin)=NaN;
%     u_filtered(u_filtered>umax)=NaN;
%     v_filtered(v_filtered<vmin)=NaN;
%     v_filtered(v_filtered>vmax)=NaN;
%     % stddev check
%     meanu=nanmean(nanmean(u_filtered));
%     meanv=nanmean(nanmean(v_filtered));
%     std2u=nanstd(reshape(u_filtered,size(u_filtered,1)*size(u_filtered,2),1));
%     std2v=nanstd(reshape(v_filtered,size(v_filtered,1)*size(v_filtered,2),1));
%     minvalu=meanu-stdthresh*std2u;
%     maxvalu=meanu+stdthresh*std2u;
%     minvalv=meanv-stdthresh*std2v;
%     maxvalv=meanv+stdthresh*std2v;
%     u_filtered(u_filtered<minvalu)=NaN;
%     u_filtered(u_filtered>maxvalu)=NaN;
%     v_filtered(v_filtered<minvalv)=NaN;
%     v_filtered(v_filtered>maxvalv)=NaN;
%     % normalized median check
%     %Westerweel & Scarano (2005): Universal Outlier detection for PIV data
%     [J,I]=size(u_filtered);
%     medianres=zeros(J,I);
%     normfluct=zeros(J,I,2);
%     b=1;
%     for c=1:2
%         if c==1; velcomp=u_filtered;else;velcomp=v_filtered;end %#ok<*NOSEM>
%         for i=1+b:I-b
%             for j=1+b:J-b
%                 neigh=velcomp(j-b:j+b,i-b:i+b);
%                 neighcol=neigh(:);
%                 neighcol2=[neighcol(1:(2*b+1)*b+b);neighcol((2*b+1)*b+b+2:end)];
%                 med=median(neighcol2);
%                 fluct=velcomp(j,i)-med;
%                 res=neighcol2-med;
%                 medianres=median(abs(res));
%                 normfluct(j,i,c)=abs(fluct/(medianres+epsilon));
%             end
%         end
%     end
%     info1=(sqrt(normfluct(:,:,1).^2+normfluct(:,:,2).^2)>thresh);
%     u_filtered(info1==1)=NaN;
%     v_filtered(info1==1)=NaN;
% 
%     typevector_filtered(isnan(u_filtered))=2;
%     typevector_filtered(isnan(v_filtered))=2;
%     typevector_filtered(typevector{PIVresult,1}==0)=0; %restores typevector for mask
%     u_filt_nointerp{PIVresult,1}=u_filtered;
%     v_filt_nointerp{PIVresult,1}=v_filtered;
%     
%     %Interpolate missing data
%     u_filtered=inpaint_nans(u_filtered,4);
%     v_filtered=inpaint_nans(v_filtered,4);
%     
%     u_filt{PIVresult,1}=u_filtered;
%     v_filt{PIVresult,1}=v_filtered;
%     typevector_filt{PIVresult,1}=typevector_filtered;
%     
% end
% 
% %disp('DONE.')
% 
% 
% 
% u_nointerp=u_filt_nointerp{1,1};
% v_nointerp=v_filt_nointerp{1,1};
% 
% u_post=u_filt{1,1};
% v_post=v_filt{1,1};
% 
% 
% u_post=(u_post*100000)/(60*60*24);  %from  [km/day] to [cm/s]
% v_post=(v_post*100000)/(60*60*24);
% 
% %mean velocities
%      u_pos=u_post>0;
%      u_neg=u_post<0;
%      A=sum(u_post.*u_pos,2)./sum(u_pos,2);  
%      B=sum(u_post.*u_neg,2)./sum(u_neg,2);  
%      nanmean(A);     %[cm/s]
%      nanmean(B);     %[cm/s]
% 
%      v_pos=v_post>0;
%      v_neg=v_post<0;
%      C=sum(v_post.*v_pos,2)./sum(v_pos,2);  
%      D=sum(v_post.*v_neg,2)./sum(v_neg,2);
%      nanmean(C);     %[cm/s]
%      nanmean(D);     %[cm/s]
%      
%      
% % mean(abs(u_post))     %after post processing
% % mean(abs(u_post))     %after post processing
% % mean(abs(u_post))     %after post processing
% % mean(abs(u_post))     %after post processing
% % 
% %      
% % min(abs(nonzeros(u_post)))     %after post processing
% % min(abs(nonzeros(v_post)))
% % mean(abs(u{1,1}))          %no post processing
% % mean(abs(v{1,1}))
% 
% 
% %weighted mean velocities
% for i=1:size(COMP,2)
%     weight(i,1)=size(COMP{1,i},1);
% end
% 
% 
% A(isnan(A))=0;
% weight_A=weight(find(A~=0));
% A1=nonzeros(A);
% 
% B(isnan(B))=0;
% weight_B=weight(find(B~=0));
% B1=nonzeros(B);
% 
% C(isnan(C))=0;
% weight_C=weight(find(C~=0));
% C1=nonzeros(C);
% 
% D(isnan(D))=0;
% weight_D=weight(find(D~=0));
% D1=nonzeros(D);
% 
% 
% 
% cd(strcat(HOME,'/input/info'))
% east=wmean(A1,weight_A)          %[cm/s]
% west=wmean(B1,weight_B)          %[cm/s]
% north=wmean(C1,weight_C)
% south=wmean(D1,weight_D)
% 
% cd(strcat(HOME,'/output/tracked'))
% save(['east'],'east');           %[cm/s]
% save(['west'],'west');
% save(['north'],'north');
% save(['south'],'south');
% 
% 
% 
% 
% %% calculate Okubo-Weiss parameterU=reshape(u_post,[g1,h1]);   %h1=dx  %g1=dy
% V=reshape(v_post,[g1,h1]);
% 
% 
% %h1=dx [pxls]->[cm]
% dx=h1*25000;
% 
% %g1=dy [pxls]->[cm]
% dy=g1*25000;
% 
% 
% % % % % % % %dudx= (diff(U(:,2:size(U,2)))-diff(U(:,1:size(U,2)-1)))./h1;
% % % % % % % dudx= diff(U)./h1;
% % % % % % % 
% % % % % % % %dv/dy
% % % % % % % dvdy= diff(V)./g1;
% % % % % % % 
% % % % % % % %vorticity: dv/dx + du/dy
% % % % % % % w= diff(V)./h1 + diff(U)./g1;
% 
% 
% 
% 
% %dudx= (diff(U(:,2:size(U,2)))-diff(U(:,1:size(U,2)-1)))./h1;
% dudx= diff(U,1,2)./dx;
% du_dx=dudx(2:end,:);
% %dv/dy
% dvdy= diff(V)./dy;
% dv_dy=dvdy(:,2:end);
% %dv/dx
% dvdx= diff(V,1,2)./dx;
% dv_dx=dvdx(2:end,:)
% %du/dy
% dudy= diff(U)./dy;
% du_dy=dudy(:,2:end);
% 
% %vorticity: dv/dx + du/dy
% w= dv_dx - du_dy;
% Sn= du_dx - dv_dy;
% Ss= dv_dx + du_dy;
% 
% %Okubo-Weiss parameter
% W=Sn.^2 + Ss.^2 - w.^2;
% W0=.2*std(std(W))
% 
% 
% 
% cmap=jet;
% X1_region=[1:h1-1];
% Y1_region=[1:g1-1];
% [xx,yy]=meshgrid(X1_region,Y1_region);
% contourf(xx,-yy,W, 'edgecolor', 'none');
% shading interp
% colormap(cmap)
% cbr=colorbar
% caxis([min(min(W)) max(max(W))])
% colormapeditor
% hold on
% 
% 
% W_grid=zeros(g1,h1);
% W_grid(2:g1,2:h1)=W;
% 
% W_grid=reshape(W_grid.',h1*g1,1);   
% 
% 
% compartment=[];
% W_grid(:,2)=1:size(W_grid,1);
% 
% for i=1:size(W_grid,1)
%     [aa,bb]=find(compartmentID==W_grid(i,2));
%     compartment(aa,bb)=W_grid(i);
% end
% 
% 
% 
% close all
% 
% contourf(X1_region',-Y1_region',compartment, 20, 'edgecolor', 'none');
% shading interp
% colormap(cmap)
% cbr=colorbar
% caxis([min(min(W)) max(max(W))])
% colormapeditor
% hold on
% 
% 
% [coast_y,coast_x]=find(im_coast==1);
% plot(coast_x,-coast_y,'w.')
% hold on
% 
% [coast_yy,coast_xx]=find(im_coast_contour==1);
% plot(coast_xx,-coast_yy,'k.')
% hold on


disp('FINISHED ROTATION')
disp('BEGIN ATM CORRELATION')


