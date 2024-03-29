
% demo of DP mixture model
dd = 2;
KK = 1;
trueK = 5;
NN = 10000;
aa = 1;
s0 = 3;
ss = 1;
numiter = 100;

hh.dd = dd;
hh.ss = s0^2/ss^2;
hh.vv = 5;
hh.VV = ss^2*eye(dd);
hh.uu = zeros(dd,1);

% construct data
truez = ceil((1:NN)/NN*trueK);
mu = randn(dd,trueK)*s0;
yy = mu(:,truez) + randn(dd,NN)*ss;
xx = num2cell(yy,1);

% initialize component assignment
zz = ceil(rand(1,NN)*KK);

% initialize DP mixture
dpm = dpm_init(KK,aa,Gaussian(hh),xx,zz);

% initialize records
record.KK = zeros(1,numiter);

% initialize colors to be used for plotting
cc = colormap;
cc = cc(randperm(size(cc,1)),:);
cc = cc(rem(1:NN,size(cc,1))+1,:);

% run
figure(1); 
tic; lasttime = toc;
for iter = 1:numiter
  % pretty pictures
  if toc>lasttime+1, 
    dpm_demo2d_plot(dpm,cc); 
    axis([min(yy(:))-1 max(yy(:))+1  min(yy(:))-1 max(yy(:))+1]);
    drawnow; 
    lasttime = toc;
    title(['DP mixture: iter# ' num2str(iter)]);
  end

  % gibbs iteration 
  dpm = dpm_gibbs(dpm,1);

  % record keeping
  record.KK(iter) = sum(dpm.nn>0);
end

