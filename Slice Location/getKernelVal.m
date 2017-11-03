function [kernelOutputs,logKernelOutputs] = getKernelVal(xpts, trXpt, bandwidth)
% 1/sqrt(2*pi)^D * exp(-z^2/2)
% output: tstDatanum x trDatanum

% [dim,datanum] = size(trXpt);
% zs = (trXpt - xpt*ones(1,datanum))/bandwidth;
% logKernelOutputs = getLogGaussian(zs, zeros(dim,1), eye(dim));
% kernelOutputs = exp(getLogGaussian(zs, zeros(dim,1), eye(dim)));


[dim,trdatanum] = size(trXpt);
[~,tstdatanum] = size(xpts);

trXptNormSqs = sum(trXpt.^2, 1);
xptNormSqs = sum(xpts.^2, 1);
logKernelOutputs = -dim/2*log(2*pi) + ...
    (-.5*(xptNormSqs'*ones(1,trdatanum) + ones(tstdatanum,1)*trXptNormSqs - 2*xpts'*trXpt))/bandwidth^2;
kernelOutputs = exp(logKernelOutputs);
