function [kernelOutputs, logKernelOutputs, logscale] = getScaledKernelVal(logkernelvals)
% scale kernel outputs
% logkernelvals: tstdatanum x trdatanum

logscale = max(logkernelvals, [], 2);
logKernelOutputs = logkernelvals - logscale*ones(1,size(logkernelvals,2));
kernelOutputs = exp(logKernelOutputs);
