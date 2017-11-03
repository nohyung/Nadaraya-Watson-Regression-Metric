function estL = getMetricForNWKernelRegression(tstPt, trYs, trData, regMultiplier)

if nargin < 4
    regMultiplier = -2;
end

[Dim,datanum] = size(trData);
% obtain estL
estMean = mean([trYs;trData],2);
estMeanx = estMean(2:end);
estSig = [trYs;trData]*[trYs;trData]'/datanum - estMean*estMean';
estSig = estSig + eye(Dim + 1)*trace(estSig)*10^-6;  % regularize both x and y
invEstSigx = inv(estSig(2:end,2:end));
Met = invEstSigx*(tstPt - estMeanx)*estSig(1,2:end)*invEstSigx;Met = Met + Met';
dim = size(Met,1);
[V,D] = eig(Met);
Evals = diag(D)';
% only two nonzero eigenvalues
[AbsEvals, sortedIndex] = sort(abs(Evals));
PosEvalIndex = [];
NegEvalIndex = [];

regR = max(AbsEvals)*10^regMultiplier;
if Evals(sortedIndex(end)) > 0
    PosEvalIndex = sortedIndex(end);
else
    NegEvalIndex = sortedIndex(end);
end
if Evals(sortedIndex(end - 1)) > 0
    PosEvalIndex = [PosEvalIndex sortedIndex(end - 1)];
else
    NegEvalIndex = [NegEvalIndex sortedIndex(end - 1)];
end
OtherIndexes = sortedIndex(1:end-2);

PosEvalNum = size(PosEvalIndex, 2);
NegEvalNum = size(NegEvalIndex, 2);

estL = [V(:,PosEvalIndex)*diag(sqrt(Evals(PosEvalIndex)*PosEvalNum + regR)) ...    % 2 ***
    V(:,NegEvalIndex)*diag(sqrt(Evals(NegEvalIndex)*NegEvalNum*(-1) + regR)) ...
    V(:,OtherIndexes)*diag(sqrt(regR))];

logDet = .5*(log(Evals(PosEvalIndex)*PosEvalNum + regR) + log(Evals(NegEvalIndex)*NegEvalNum*(-1) + regR) + length(OtherIndexes)*log(regR));
estL = estL/exp(logDet/dim);

