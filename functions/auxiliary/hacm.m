function [ybar,vybar] = hacm(y,nma,ikern);
% Sample mean and hac var for mean
%  nma = number of MA terms in kernel
%  ikern = 0 (flat kernel)
%        = 1  Bartlett kernel
%  
ybar=mean(y);
z = y - repmat(ybar,size(y,1),1);

v = zeros(size(z,2));
% Form Kernel 
kern=zeros(nma+1,1);
for ii = 0:nma;
    kern(ii+1,1)=1;
    if nma > 0;
       if ikern == 1; 
           kern(ii+1,1)=(1-(ii/(nma+1))); 
       end;
    end;
end;

%Form Hetero-Serial Correlation Robust Covariance Matrix 
for ii = -nma:nma;
  if ii <= 0; 
      r1=1; 
      r2=size(z,1)+ii; 
  else; 
      r1=1+ii; 
      r2=size(z,1); 
  end;
  v = v + kern(abs(ii)+1,1)*(z(r1:r2,:)'*z(r1-ii:r2-ii,:));
end;
v = v/(size(z,1)-1);
vybar=v/size(z,1);

end

