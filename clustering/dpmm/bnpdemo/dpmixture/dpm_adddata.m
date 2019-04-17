function dpm = dpm_adddata(dpm,xx)
% add data points xx into DP mixture model.  
% xx data, x_i=xx{i}
% will add each data item into its own cluster.

nx = numel(xx);
KK = dpm.KK;
dpm.KK = dpm.KK + nx;
dpm.NN = dpm.NN + nx;
dpm.qq(end+(1:nx)) = dpm.qq(end);
dpm.xx(end+(1:nx)) = xx(:);
dpm.zz(end+(1:nx)) = KK+(1:nx);
dpm.nn(end+(1:nx)) = 1;

% add data items into mixture components
for ii = 1:nx
  kk = KK+ii;
  dpm.qq{kk} = additem(dpm.qq{kk},xx{ii});
end

