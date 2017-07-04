
testSamples = find(imdb.images.set == 2);
results = zeros(256,256,1,length(testSamples));
groundTruth = imdb.images.labels(:,:,:,testSamples);

for idx = 1:length(testSamples)
    tic;
    testIDX = testSamples(idx);
    imageIVUSTest = single(imdb.images.data(:,:,:,testIDX));
    imageTest_GT = single(imdb.images.labels(:,:,:,testIDX));

    numJitter = 10;
    imageTest_M1 = permute(repmat(imageIVUSTest,[1,1,numJitter]),[1,2,4,3]);
    currentImage = padarray(imageIVUSTest,[32, 32],0,'both');
    rotationVals =  [0; -45 + 90*rand(9,1)];

    idxAug =1;

    for rotationIDX = 1:length(rotationVals)
        currentImageRotated = imrotate(currentImage,rotationVals(rotationIDX),'nearest','crop');
        currentImageRotated = currentImageRotated(33:end-32,33:end-32,:,:);
        imageTest_M1(:,:,:,idxAug) = gather(currentImageRotated);
        idxAug = idxAug +1;
    end  

     opts.border = [8 8 8 8]; % tblr

    % augmenting data - M1 - Jitter and Fliplr
    augData_M1 = zeros(size(imageTest_M1) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', imageTest_M1);

    augData_M1(opts.border(1)+1:end-opts.border(2), ...
    opts.border(3)+1:end-opts.border(4), :, :) = imageTest_M1;




    sz0 = size(augData_M1);
    sz = size(imageTest_M1);
    images_M1 = imageTest_M1; 
    locArray = zeros(numJitter,2);
    for idxJitter = 1:numJitter
        loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
        images_M1(:,:,:,idxJitter) = augData_M1(loc(1):loc(1)+sz(1)-1, ...
        loc(2):loc(2)+sz(2)-1, :, idxJitter); 
        locArray(idxJitter,:) = loc;
    end

    net.eval({'input_M1', gpuArray(images_M1)});
    segmentationPlaque = net.vars(net.getVarIndex('sigmoid1')).value;
    segmentationPlaque = gather(segmentationPlaque);

    % augmenting reconstruction 
    augRecon_L1 = zeros(size(segmentationPlaque) + [sum(opts.border(1:2)) ...
    sum(opts.border(3:4)) 0 0], 'like', segmentationPlaque);



    for idxJitter = 1:numJitter
        loc = locArray(idxJitter,:);
        augRecon_L1(loc(1):loc(1)+sz(1)-1, ...
        loc(2):loc(2)+sz(2)-1, :, idxJitter) = segmentationPlaque(:,:,:,idxJitter);
    end

    segmentationPlaque = augRecon_L1(opts.border(1):sz0(1)-opts.border(2)-1,opts.border(3):sz0(2)-opts.border(4)-1,:,:);

    rotationVals = -1*rotationVals;

    idxAug =1;

    for rotationIDX = 1:length(rotationVals)
        currentImage = segmentationPlaque(:,:,:,rotationIDX);
        currentImage = padarray(currentImage,[32, 32],0,'both');

        currentImageRotated = imrotate(currentImage,rotationVals(rotationIDX),'nearest','crop');
        currentImageRotated = currentImageRotated(33:end-32,33:end-32,:,:);
        segmentationPlaque(:,:,:,idxAug) = gather(currentImageRotated);
        idxAug = idxAug +1;
    end 




    results(:,:,:,idx) = median(segmentationPlaque,4);

    toc;
end
% figure, 
% subplot(1,3,1), imshow(mat2gray(squeeze(imageIVUSTest))), title('Input IVUS');
% subplot(1,3,2), imshow(squeeze(imageTest_GT)), title('Ground Truth Segmentation');
% subplot(1,3,3), imshow(mat2gray(squeeze(segmentationPlaqueFilt))), title('Predicted Segmentation');
