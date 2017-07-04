function Y = vl_nnlossAE(X,c, do_weights, dzdy)
% VL_NNLOSS  CNN log-loss
%    Y = VL_NNLOSS(X, C) applies the the logistic loss to the data
%    X. X has dimension H x W x D x N, packing N arrays of W x H
%    D-dimensional vectors.
%
%    C contains the class labels, which should be integers in the range
%    1 to D. C can be an array with either N elements or with dimensions
%    H x W x 1 x N dimensions. In the fist case, a given class label is
%    applied at all spatial locations; in the second case, different
%    class labels can be specified for different locations.
%
%    DZDX = VL_NNLOSS(X, C, DZDY) computes the derivative DZDX of the
%    function projected on the output derivative DZDY.
%    DZDX has the same dimension as X.



% no division by zero
X = X + 1e-4 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;


res=(c-X);

n=1;
if isempty(dzdy) %forward
    Y = (sum(res(:).^2))/numel(res);
else
    Y_= -1.*(c-X);
    Y = single (Y_ * (dzdy / n) );
end




end
