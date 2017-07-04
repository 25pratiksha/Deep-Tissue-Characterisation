function [net, info] = DeconvNetAE_CIFAR_L2_4Blocks(imdb, netF, inpt, varargin)

	% some common options
	trainer = @cnn_train_dag_ConvAE;

	opts.train.extractStatsFn = @extract_stats_segmentationAE;
	opts.train.batchSize = 200;
	opts.train.numEpochs = 60 ;
	opts.train.continue = true ;
	opts.train.gpus = [1] ;
	%opts.train.learningRate = [1e-4*ones(1, 20),  1e-5*ones(1, 20),1e-6*ones(1, 20)];
	opts.train.learningRate = 1e-6;
    opts.train.weightDecay = 1e-4;
	opts.train.momentum = 0.9;
	opts.train.expDir = inpt.expDir;
	opts.train.savePlots = false;
	opts.train.numSubBatches = 1;
	% getBatch options
	bopts.useGpu = numel(opts.train.gpus) >  0 ;
    
    
	% organize data
	K = 2; % how many examples per domain	
	trainData = find(imdb.images.set == 1);
	valData = find(imdb.images.set == 3);
	
	% debuging code
	opts.train.exampleIndices = [trainData(randperm(numel(trainData), K)), valData(randperm(numel(valData), K))];

	
	% network definition
	net = dagnn.DagNN() ;
    % Encoder Block1
	net.addLayer('conv1', dagnn.Conv('size', [7 7 3 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'input'}, {'conv1'},  {'conv1f'  'conv1b'});
	net.addLayer('bn1', dagnn.BatchNorm('numChannels', 64), {'conv1'}, {'bn1'}, {'bn1f', 'bn1b', 'bn1m'});
	net.addLayer('relu1', dagnn.ReLU(), {'bn1'}, {'relu1'}, {});
	net.addLayer('pool1', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu1'}, {'pool1'}, {});

    % Encoder Block 2
	net.addLayer('conv2', dagnn.Conv('size', [7 7 64 64], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool1'}, {'conv2'},  {'conv2f'  'conv2b'});
	net.addLayer('bn2', dagnn.BatchNorm('numChannels', 64), {'conv2'}, {'bn2'}, {'bn2f', 'bn2b', 'bn2m'});
	net.addLayer('relu2', dagnn.ReLU(), {'bn2'}, {'relu2'}, {});
	net.addLayer('pool2', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu2'}, {'pool2'}, {});

    % Encoder Block 3
    net.addLayer('conv3', dagnn.Conv('size', [3 3 64 128], 'hasBias', true, 'stride', [1, 1], 'pad', [3 3 3 3]), {'pool2'}, {'conv3'},  {'conv3f'  'conv3b'});
	net.addLayer('bn3', dagnn.BatchNorm('numChannels', 128), {'conv3'}, {'bn3'}, {'bn3f', 'bn3b', 'bn3m'});
	net.addLayer('relu3', dagnn.ReLU(), {'bn3'}, {'relu3'}, {});
	net.addLayer('pool3', dagnn.Pooling('method', 'max', 'poolSize', [2, 2], 'stride', [2, 2], 'pad', [0 0 0 0]), {'relu3'}, {'pool3'}, {});
    
    % Fully Connected Block
    
    net.addLayer('fc', dagnn.Conv('size', [1 1 128 512], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'pool3'}, {'fcOut'},  {'fcf'  'fcb'});
	net.addLayer('reluFC', dagnn.ReLU(), {'fcOut'}, {'reluFC'}, {});
    
    % Decoder Block 4
    
    net.addLayer('deconv4', dagnn.ConvTranspose('size', [3 3 64 512], 'hasBias', true, 'upsample', [2,2], 'crop', [3 2 3 2]), {'reluFC'}, {'deconv4'},  {'deconv4f'  'deconv4b'});
	net.addLayer('bn4', dagnn.BatchNorm('numChannels', 64), {'deconv4'}, {'bn4'}, {'bn4f', 'bn4b', 'bn4m'});
	net.addLayer('relu4', dagnn.ReLU(), {'bn4'}, {'relu4'}, {});
    
    
	net.addLayer('deconv5', dagnn.ConvTranspose('size', [7 7 64 64], 'hasBias', true, 'upsample', [2,2], 'crop', [3 2 3 2]), {'relu4'}, {'deconv5'},  {'deconv5f'  'deconv5b'});
	net.addLayer('bn5', dagnn.BatchNorm('numChannels', 64), {'deconv5'}, {'bn5'}, {'bn5f', 'bn5b', 'bn5m'});
	net.addLayer('relu5', dagnn.ReLU(), {'bn5'}, {'relu5'}, {});


	net.addLayer('deconv6', dagnn.ConvTranspose('size', [7 7 64 64], 'hasBias', true, 'upsample', [2,2], 'crop', [3 2 3 2]), {'relu5'}, {'deconv6'},  {'deconv6f'  'deconv6b'});
	net.addLayer('bn6', dagnn.BatchNorm('numChannels', 64), {'deconv6'}, {'bn6'}, {'bn6f', 'bn6b', 'bn6m'});
	net.addLayer('relu6', dagnn.ReLU(), {'bn6'}, {'relu6'}, {});

	net.addLayer('reconstruction', dagnn.Conv('size', [1 1 64 3], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu6'}, {'reconstruction'},  {'classf'  'classb'});
	net.addLayer('objective', dagnn.LossAE('weights', 'true'), {'reconstruction','label'}, {'objective'});
	% -- end of the network

% 	% do the training!
%     if ~isempty(netF)
% 		%initNet_FineTuning(net, netF);
% 	else
% 		initNet(net, 1e-2*ones(1, 8), 1e-2*ones(1, 8));
% 	end
	initNet(net);
	net.conserveMemory = false;

	info = trainer(net, imdb, @(i,b) getBatch(bopts,i,b), opts.train, 'train', trainData, 'val', valData) ;
end


% function on charge of creating a batch of images + labels
function inputs = getBatch(opts, imdb, batch)
	images = imdb.images.data(:,:,:,batch) ;
	labels = imdb.images.labels(:, :, :, batch) ;
	if opts.useGpu > 0
  		images = gpuArray(images);
		labels = gpuArray(labels);
	end

	inputs = {'input', images, 'label', labels} ;
end

function initNet(net)
	net.initParams();
	%
	i = 1;
	for l=1:length(net.layers)
		% is a convolution layer?
		if(strcmp(class(net.layers(l).block), 'dagnn.Conv') || strcmp(class(net.layers(l).block), 'dagnn.ConvTranspose'))
			f_ind = net.layers(l).paramIndexes(1);
			b_ind = net.layers(l).paramIndexes(2);

			[h,w,in,out] = size(net.params(f_ind).value);
			xavier_gain = 0.7*sqrt(1/(size(net.params(f_ind).value,1)*size(net.params(f_ind).value,2)*size(net.params(f_ind).value,3))) % sqrt(1/fan_in)
			net.params(f_ind).value = xavier_gain*randn(size(net.params(f_ind).value), 'single');
			net.params(f_ind).learningRate = 1;
			net.params(f_ind).weightDecay = 1;

			net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
			net.params(b_ind).learningRate = 0.5;
			net.params(b_ind).weightDecay = 1;
            
            
            
            
%             f_ind = net.layers(l).paramIndexes(1);
% 			b_ind = net.layers(l).paramIndexes(2);
% 
% 			net.params(f_ind).value = F(i)*randn(size(net.params(f_ind).value), 'single');
% 			net.params(f_ind).learningRate = 1;
% 			net.params(f_ind).weightDecay = 1;
% 
% 			net.params(b_ind).value = B(i)*randn(size(net.params(b_ind).value), 'single');
% 			net.params(b_ind).learningRate = 0.5;
% 			net.params(b_ind).weightDecay = 1;
% 			i = i + 1;
		end
	end
end


	


