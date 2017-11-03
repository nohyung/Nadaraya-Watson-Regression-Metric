function Risks = getLoss_fast2_gpu(curL, trData, trYs, hbandwidths)
% tic;Risks = getLoss_fast2_gpu(curL, trData_gpu, trYs_gpu, hbandwidths);toc
% Compare with
% tic;Risks = getLoss_fast2(curL, trData, trYs, hbandwidths);toc


numBandWidths = size(hbandwidths,2);
[Dim, datanum] = size(trData);
predYs = zeros(numBandWidths, datanum, 'gpuArray');
transformedTrData = curL'*trData;
ones_xptNormSqs = ones(datanum,1)*sum(transformedTrData.^2, 1);
distanceSqMat = ones_xptNormSqs' + ones_xptNormSqs - 2*transformedTrData'*transformedTrData;
for icntBandwidth = 1:numBandWidths
    curBandwidth = hbandwidths(icntBandwidth);
%     [~, totlogKernelOutputs] = getKernelVal(transformedTrData, transformedTrData, curBandwidth);
%     totlogKernelOutputs = getLogKernelVal(transformedTrData, transformedTrData, curBandwidth);
%     totlogKernelOutputs = getSymmetricLogKernelVal(transformedTrData, curBandwidth);
    totlogKernelOutputs = -Dim/2*log(2*pi*curBandwidth^2) + -.5*distanceSqMat/curBandwidth^2;
    totlogKernelOutputs = totlogKernelOutputs - diag(max(max(abs(totlogKernelOutputs), [], 1), [], 2)*ones(1,datanum));  % fill diagonal with minimum
    kernelOutputs = getScaledKernelVal(totlogKernelOutputs);    % diagonal component is zero, and the others are scaled without diagonal element
    kernelOutputs = kernelOutputs - diag(diag(kernelOutputs));  % put diagonal as zeros
    predYs(icntBandwidth, :) = (kernelOutputs*trYs')./sum(kernelOutputs, 2);   % tstdatanum x 1
end

Risks = mean((predYs - ones(numBandWidths,1)*trYs).^2, 2)';
% Risks

