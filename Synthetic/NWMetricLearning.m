function NWMetricLearning()

iternum = 100;
hbandwidths = exp([-1.7:.2:4.3]);

Dim = 10;

ySigSq = .1;
datanum = 100;

predYs = zeros(iternum, size(hbandwidths,2));
predYsMetric = zeros(iternum, size(hbandwidths,2));
for iiter = 1:iternum
    printIterNum = 1000;
    if floor(iiter/printIterNum) == iiter/printIterNum
        sprintf('iiter[%d]', iiter)
    end
    %%%%%%%%%%%%%%%% Gaussian x %%%%%%%%%%%%%%%
    mux = zeros(Dim,1);
    Sigx = eye(Dim);
    trData = mvnrnd(mux', Sigx, datanum)';
    tstPtAngle = 30;    % good for 3-D - 1
    tstPt = 1*[cos(tstPtAngle/180*pi) sin(tstPtAngle/180*pi) zeros(1,Dim - 2)]';
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Wvec = [5;zeros(Dim - 1,1)];    % good for 3-D - 1
    yIntersect = 0;
    trYs = Wvec'*trData + yIntersect + randn(1,datanum)*sqrt(ySigSq);
    trueTstYs = Wvec'*tstPt + yIntersect;
    tstYs = trueTstYs + randn(1)*sqrt(ySigSq);

    for ibandWidthIdx = 1:size(hbandwidths,2)
        hbandwidth = hbandwidths(ibandWidthIdx);
        
        [kernelOutputs, logKernelOutputs] = getKernelVal(tstPt, trData, hbandwidth);
        kernelOutputs = getScaledKernelVal(logKernelOutputs);    % [kernelOutputs, logKernelOutputs, logscale] = getScaledKernelVal(logKernelOutputs);
        predYs(iiter,ibandWidthIdx) = (kernelOutputs*trYs')./sum(kernelOutputs, 2);   % tstdatanum x 1
        
        % With Metric
        regMultiplier = -5;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % In the experiments with real data, "regMultiplier = -2" performs
        % the best. We used "regMultiplier = -2" for our all benchmark data
        % experiments.
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        estL = getMetricForNWKernelRegression(tstPt, trYs, trData, regMultiplier);
        [kernelOutputs, logKernelOutputs] = getKernelVal(estL'*tstPt, estL'*trData, hbandwidth);
        kernelOutputs = getScaledKernelVal(logKernelOutputs);    % [kernelOutputs, logKernelOutputs, logscale] = getScaledKernelVal(logKernelOutputs);
        predYsMetric(iiter,ibandWidthIdx) = (kernelOutputs*trYs')./sum(kernelOutputs, 2);   % tstdatanum x 1

    end

end

BiasesToTrueYs = mean(predYs - trueTstYs, 1);
StdErrBiasesToTrueYs = std(predYs - trueTstYs, 1, 1)/sqrt(iternum);
StdErrBiasesToTrueYsSq = std(predYs - trueTstYs, 1, 1).^2/sqrt(iternum);
Biases = mean(predYs - tstYs, 1);
StdErrBiases = std(predYs - tstYs, 1, 1)/sqrt(iternum);
MSEToTrueTstYs = mean((predYs - trueTstYs).^2, 1);
StdErrMSEsToTrueTstYs = std(predYs - trueTstYs, 1, 1)/sqrt(iternum);
MSEs = mean((predYs - tstYs).^2, 1);
StdErrMSEs = std(predYs - tstYs, 1, 1)/sqrt(iternum);

BiasesToTrueYsMetric = mean(predYsMetric - trueTstYs, 1);
StdErrBiasesToTrueYsMetric = std(predYsMetric - trueTstYs, 1, 1)/sqrt(iternum);
StdErrBiasesToTrueYsSqMetric = std(predYsMetric - trueTstYs, 1, 1).^2/sqrt(iternum);
BiasesMetric = mean(predYsMetric - tstYs, 1);
StdErrBiasesMetric = std(predYsMetric - tstYs, 1, 1)/sqrt(iternum);
MSEToTrueTstYsMetric = mean((predYsMetric - trueTstYs).^2, 1);
StdErrMSEsToTrueTstYsMetric = std(predYsMetric - trueTstYs, 1, 1)/sqrt(iternum);
MSEsMetric = mean((predYsMetric - tstYs).^2, 1);
StdErrMSEsMetric = std(predYsMetric - tstYs, 1, 1)/sqrt(iternum);


figure
hold on

plot(log(hbandwidths), MSEToTrueTstYs, 'bo-', 'LineWidth', 1.5)
plot(log(hbandwidths), MSEToTrueTstYsMetric, 'k*-', 'LineWidth', 1.5)
legend('Location', 'East', 'MSE', 'MSE with Metric')
shadedErrorBar(log(hbandwidths), MSEToTrueTstYs, StdErrMSEsToTrueTstYs, 'bo-', 1);
shadedErrorBar(log(hbandwidths), MSEToTrueTstYsMetric, StdErrMSEsToTrueTstYsMetric, 'k*-', 1);

set(gca, 'FontSize', 20)
xlabel('log(h)', 'FontSize', 25)
ylabel('MSE', 'FontSize', 25)

grid on
axis([-2 4.5 -1 17])
print -dpng NWMetricLearning_10D

