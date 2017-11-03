%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function logrho = getLogGaussian(x, mu, Sig)
% Src Code: 1401050001

datasize = size(x, 2);
evals = eig(Sig)';
dim = size(mu,1);
logrho = zeros(1,datasize);
sumlogevals = sum(log(evals), 2);
invSig = inv(Sig);
preSum = -dim/2*log(2*pi) - .5*sumlogevals;
for icnt = 1:datasize
    dx = (x(:,icnt) - mu);
    logrho(1,icnt) =  preSum - .5*dx'*invSig*dx;
end

