function stats = extract_stats_segmentationAE(net, stats)
	sel = find(cellfun(@(x) isa(x,'dagnn.LossAE'), {net.layers.block})) ;

    obj = net.vars(net.layers(sel).outputIndexes(1)).value;
     if(isfield(stats, 'objective'))
        stats.objective = obj + stats.objective;
    else
        stats.objective = obj;
     end
end
