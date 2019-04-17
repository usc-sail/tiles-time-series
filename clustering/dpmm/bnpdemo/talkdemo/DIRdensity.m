% plots out heatmap of density for Dirichlet distribution on 3-simplex

% dirichlet parameters
AA = [1 1 1; 2 2 2; 5 5 5; 5 5 2; 5 2 2; .7 .7 .7];

% mesh grating
nn = 1001; % number of points at which to evaluate
tiny = .0001; % bound away from zero 
dd = 1/(nn-1);

% equilateral triangle
MM = [cos(pi/3) 1 0; sin(pi/3) 0 0; 1 1 1];
IM = inv(MM);
x1 = reshape((0:dd:1)'   * ones(1,nn), 1, nn*nn);
y1 = reshape(ones(1,nn)' * (0:dd:1)  , 1, nn*nn);
z1 = reshape(ones(1,nn)' * ones(1,nn), 1, nn*nn);
XX = IM * [x1; y1; z1];
xx = XX(1,:);
yy = XX(2,:);
zz = XX(3,:);

pp = zeros(1,nn*nn);
for run = 1:size(AA,1)

  aa = AA(run,:);

  % log normalization constant
  lz = gammaln(sum(aa))-sum(gammaln(aa));
  for ii = 1:nn*nn
    if xx(ii) > tiny & yy(ii) >tiny & zz(ii) > tiny
      % density
      pp(ii) = -exp(log(xx(ii))*(aa(1)-1)+log(yy(ii))*(aa(2)-1)...
          +log(zz(ii))*(aa(3)-1)+lz);
    else
      pp(ii) = 0;
    end
  end

  % display heatmap of density
  colormap hot
  imagesc(reshape(pp,[nn nn]),[-10 0]);
  hold on
  plot([1.5 1.5 sqrt((nn-1)^2-(nn/2)^2)+.5 1.5],[.5 (nn-.5) (nn/2+.5) 0.5],'k');
  hold off
  axis([-1 sqrt((nn-1)^2-(nn/2)^2)+.5 -1 nn])
  axis equal
  axis off
  title(sprintf('Dir(%.1f,%.1f,%.1f)',aa(1),aa(2),aa(3)));
  % print('-depsc','-zbuffer',...
  %      ['dirc-' num2str(aa(1)) '-' num2str(aa(2)) '-' num2str(aa(3)) '.eps']);
  pause
end



