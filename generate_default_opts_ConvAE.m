function opts = generate_default_opts_ConvAE()
	opts.expDir = '';
	opts.continue = true ;
	opts.batchSize = 256 ;
	opts.numSubBatches = 1 ;
	opts.train = [] ;
	opts.val = [] ;
	opts.gpus = [1] ;
	opts.prefetch = false ;
	opts.numEpochs = 300 ;
	opts.learningRate = 0.001 ;
	opts.weightDecay = 0.0005 ;
	opts.momentum = 0.9 ;
	opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
	opts.profile = false ;
	opts.savePlots = true;

	% stats and visualization
    opts.exampleIndices = [];
	opts.derOutputs = {'objective', 100} ;
	opts.extractStatsFn = @extract_stats_reconstruction;
	opts.fcn_visualize = @visualize_reconstruction;
    opts.vis.hnd_loss = figure()
	opts.vis.hand_examples = figure()
	% ----
end
