function RegressSlice_Localization()

% numWorkers = 8;
% pool = parpool(numWorkers);

hbandwidths = exp([-8:.8:3]);

%%%
load slice_localization_data    % X Y, 384x53500

realizeNum = 5;
% realizeNum = 20;   % We used 'realizeNum = 20' for our experiments in the paper

totalData = X;
totalTarget = Y;
[Dim,totalDataNum] = size(totalData);

trDataNum = 5*10^3;
tstDataNum = 7*10^2;
TrIdxes = zeros(realizeNum, trDataNum);
TstIdxes = zeros(realizeNum, tstDataNum);
for irealize = 1:realizeNum
    newIndex = randperm(totalDataNum);
    TrIdxes(irealize,:) = newIndex(1:trDataNum);
    TstIdxes(irealize,:) = newIndex(trDataNum+[1:tstDataNum]);
end

tic
TestPredYs = zeros(realizeNum, tstDataNum);   % Nadaraya-Watson
TestPredYs3 = zeros(realizeNum, tstDataNum);  % Nadaraya-Watson metric learning with estimated parameters
tstDataTargets = zeros(realizeNum, tstDataNum);   % true targets
minBandWidthIdxesNWKR = zeros(1,realizeNum);
minBandWidthIdxesNWKRMetric = zeros(1,realizeNum);
for irealize = 1:realizeNum
    trData = totalData(:,TrIdxes(irealize,:));
    trDataTarget = totalTarget(TrIdxes(irealize,:));
    % Divide data into train and validation sets
    ValidateRatio = .3;
    randPermIdxes = randperm(trDataNum);
    DataNumForValidation = floor(trDataNum*ValidateRatio);
    DataForLearn = trData(:,randPermIdxes(1:(trDataNum - DataNumForValidation)));
    TargetForLearn = trDataTarget(randPermIdxes(1:(trDataNum - DataNumForValidation)));
    DataForValidate = trData(:,randPermIdxes((trDataNum - DataNumForValidation) + 1:end));
    TargetForValidate = trDataTarget(randPermIdxes((trDataNum - DataNumForValidation) + 1:end));

    KernelMetricregMultiplier = -2;   % We used this regularization constant for all benchmark dataset experiments.

    PredYs = zeros(size(hbandwidths, 2), DataNumForValidation);   % Nadaraya-Watson
    PredYs3 = zeros(size(hbandwidths, 2), DataNumForValidation);  % Nadaraya-Watson metric learning with estimated parameters
    for ibandWidthIdx = 1:size(hbandwidths,2)
        hbandwidth = hbandwidths(ibandWidthIdx);

        %%%%%%%%%%%%%%%%%%%%%%%%%% kernel regression %%%%%%%%%%%%%%
        [kernelvals,logkernelvals] = getKernelVal(DataForValidate, DataForLearn, hbandwidth);

        maxlogkernelvals = max(logkernelvals, [], 2);
        predY = sum(exp(logkernelvals - maxlogkernelvals*ones(1,trDataNum - DataNumForValidation))*TargetForLearn', 2)./ ...
            sum(exp(logkernelvals - maxlogkernelvals*ones(1,trDataNum - DataNumForValidation)), 2);

        PredYs(ibandWidthIdx,:) = predY;    % result for validate data
        
        sprintf('NWKR finished RealizeNum[%d] BandWidth Idx[%d] Time [%d]', irealize, ibandWidthIdx, toc)

        %%%%%%%%%%%%%%%%%%%%%%%%%% Metric kernel density regression from estimated parameters  %%%%%%%%%%%%%%
        for idata = 1:DataNumForValidation
            tstPt = DataForValidate(:,idata);
            estL = getMetricForNWKernelRegression(tstPt, TargetForLearn, DataForLearn, KernelMetricregMultiplier);
            [kernelvals,logkernelvals] = getKernelVal(estL'*tstPt, estL'*DataForLearn, hbandwidth);

            maxlogkernelvals = max(logkernelvals);
            predY = sum(exp(logkernelvals - maxlogkernelvals).*TargetForLearn, 2)./sum(exp(logkernelvals - maxlogkernelvals), 2);
            PredYs3(ibandWidthIdx, idata) = predY;
        end

        sprintf('NWKR with Metric finished RealizeNum[%d] BandWidth Idx[%d] Time [%d]', irealize, ibandWidthIdx, toc)
    end
    MSENWKR = mean((PredYs - ones(size(hbandwidths,2),1)*TargetForValidate).^2, 2);
    MSENWKRMetric = mean((PredYs3 - ones(size(hbandwidths,2),1)*TargetForValidate).^2, 2);    
    varTarget = var(TargetForValidate, 1);
    NMSENWKR = MSENWKR/varTarget;
    NMSENWKRMetric = MSENWKRMetric/varTarget;
    [minBandwidthNWKR, minBandwidthIdxNWKR] = min(NMSENWKR);
    [minBandwidthNWKRMetric, minBandwidthIdxNWKRMetric] = min(NMSENWKRMetric);

    minBandwidthNWKR
    minBandwidthNWKRMetric
    minBandwidthIdxNWKR
    minBandwidthIdxNWKRMetric

    minBandWidthIdxesNWKR(irealize) = minBandwidthIdxNWKR;
    minBandWidthIdxesNWKRMetric(irealize) = minBandwidthIdxNWKRMetric;

    %% Now, Apply to Test Data
    tstData = totalData(:,TstIdxes(irealize,:));
    tstDataTarget = totalTarget(TstIdxes(irealize,:));

    hbandwidth = hbandwidths(minBandwidthIdxNWKR);
    %%%%%%%%%%%%%%%%%%%%%%%%%% kernel regression %%%%%%%%%%%%%%
    [kernelvals,logkernelvals] = getKernelVal(tstData, trData, hbandwidth);
    
    maxlogkernelvals = max(logkernelvals, [], 2);
    predY = sum(exp(logkernelvals - maxlogkernelvals*ones(1,trDataNum))*trDataTarget', 2)./ ...
        sum(exp(logkernelvals - maxlogkernelvals*ones(1,trDataNum)), 2);
    TestPredYs(irealize,:) = predY;
    sprintf('NWKR TEST finished RealizeNum[%d] Time [%d]', irealize, toc)
    
    hbandwidth = hbandwidths(minBandwidthIdxNWKRMetric);
    %%%%%%%%%%%%%%%%%%%%%%%%%% Metric kernel density regression from estimated parameters  %%%%%%%%%%%%%%
    for idata = 1:tstDataNum
        tstPt = tstData(:,idata);
        estL = getMetricForNWKernelRegression(tstPt, trDataTarget, trData, KernelMetricregMultiplier);
        [kernelvals,logkernelvals] = getKernelVal(estL'*tstPt, estL'*trData, hbandwidth);
        
        maxlogkernelvals = max(logkernelvals);
        predY = sum(exp(logkernelvals - maxlogkernelvals).*trDataTarget, 2)./sum(exp(logkernelvals - maxlogkernelvals), 2);
        TestPredYs3(irealize,idata) = predY;
    end

    sprintf('NWKR TEST with Metric finished RealizeNum[%d] Time [%d]', irealize, toc)


    tstDataTargets(irealize,:) = tstDataTarget;
end


% tstDataTarget
MSENWKRs = mean((TestPredYs - tstDataTargets).^2, 2);
MSENWKRMetrics = mean((TestPredYs3 - tstDataTargets).^2, 2);
varTargets = var(tstDataTargets, 1, 2);
NMSENWKRs = MSENWKRs./varTargets;
NMSENWKRMetrics = MSENWKRMetrics./varTargets;
NMSENWKR = mean(NMSENWKRs);
NMSENWKRMetric = mean(NMSENWKRMetrics);
stdErrNMSENWKR = std(NMSENWKRs, 1)/sqrt(realizeNum);
stdErrNNWKRMetric = std(NMSENWKRMetrics, 1)/sqrt(realizeNum);


minBandWidthIdxesNWKR
minBandWidthIdxesNWKRMetric
NMSENWKR
NMSENWKRMetric

save RegressSliceLocalizationResult MSENWKRs MSENWKRMetrics ...
    varTargets NMSENWKRs NMSENWKRMetrics ...
    NMSENWKR NMSENWKRMetric ...
    stdErrNMSENWKR stdErrNNWKRMetric ...
    minBandWidthIdxesNWKR minBandWidthIdxesNWKRMetric ...
    TestPredYs TestPredYs3 tstDataTargets

% delete(pool);
