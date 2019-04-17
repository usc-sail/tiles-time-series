% demo of DP mixture model in 1D
dd = 1;  % dimension
KK = 0;  % number of initial clusters
NN = 0;  % start empty dataset
xx = []; % start empty dataset
aa = 1;  % alpha parameter
numiter = 1000;
decay = .04;

s0 = 6;  % hyperparameter: rough guess of std of cluster means
ss = 2;  % hyperparameter: rough guess of std of cluster variances
hh.dd = dd; % dimensionality
hh.ss = s0^2/ss^2; % variance of cluster means / cluster variances
hh.vv = 1.5; % degrees of freedom of inverse wishart prior over cluster variances
hh.VV = ss^2*eye(dd); % mean cluster variances
hh.uu = zeros(dd,1); % mean of cluster means


% set range over which to evaluate density
axisrange = [-15 15 0 .5];
yy = -15:.01:15;

% construct DP mixture with no data items to get prior samples of densities
dpm = dpm_init(0,aa,Gaussian(hh),{},[]);

% run

density = dpm_demo1d_density(dpm,yy); 
avgdensity = decay*dpm_demo1d_density(dpm,yy); 
ah = plot(yy,avgdensity,'k-','linewidth',2); hold on
dh = plot(yy,density,'b-','linewidth',2); hold off
title('Samples from the posterior density of a DP mixture model; click to add points');
axis(axisrange);
pointq_setup(gca);

while (1)
  tic
  while toc < .2
    % gibbs iteration 
    dpm = dpm_gibbs(dpm,1);
  end
  % plot density
  density = dpm_demo1d_density(dpm,yy); 
  avgdensity = (1-decay)*avgdensity+decay*density;
  set(dh,'ydata',density);
  set(ah,'ydata',avgdensity);
  drawnow;
  % get new points from queue, if any
  qq = get(gca,'userdata');
  set(gca,'userdata',zeros(0,2));
  if ~isempty(qq)
    lh = line(qq(:,1),.1*ones(1,size(qq,1)));
    set(lh,'linestyle','x','color',[1 0 0]);
    dpm = dpm_adddata(dpm,num2cell(qq(:,1)'));
  end
end

