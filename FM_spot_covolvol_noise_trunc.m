function [CVV_spot,VV1_spot,VV2_spot,tau] = FM_spot_covolvol_noise_trunc(x1,x2,t1,t2,tau,T)

const=2*pi/T;

% Make sure that x1,x2,t1,t2 and tau are column vectors.
x1=x1(:);
t1=t1(:);
x2=x2(:);
t2=t2(:);
tau=tau(:);

r=diff(x1);  
rb=diff(x2);  

n1=length(r); n2=length(rb);
N=floor(10*min(n1,n2)^0.5 );
M=floor((1/3)*N^0.5);
L=floor(M^0.5);

c1=abs(r)<0.5*(T/n1)^(1/2);
c2=abs(rb)<0.5*(T/n2)^(1/2);

r=r.*c1;
rb=rb.*c2; 

S=-L-M-N:1:N+M+L;
S1=1:1:N+M+L;
 
%%  Estimation of log-ret coeffs


 
c_r1=zeros(length(S1),1); 
tt1=-1i*const*t1(1:end-1);

for j = 1:  length(S1)  
     
c_r1(j)= exp(S1(j)*tt1).'*r;  
    
end 
 
c_r=1/T* [flip(conj(c_r1)); sum(r); c_r1];




%% estimation of vol coeffs
 


c_s=zeros(2*M+1,1); c_ds=zeros(2*M+1,1); c_r2=zeros(2*M+1,2*N+1);
 

for j = 1 : 2*M+1
c_r2(j,1:2*N+1)=   c_r(ceil(length(S)/2)-N+(j-M-1) : ceil(length(S)/2)+N+(j-M-1));    
c_s(j)=T/(2*N+1)*c_r(ceil(length(S)/2)-N : ceil(length(S)/2)+N).'*flip(c_r2(j,:)).';
c_ds(j)=const*1i*(j-M-1 )*c_s(j);
 
end


 
%% estimation of volvol coeffs


 
MM=M+L;
 
c_s4=zeros(2*MM+1,1); c_r22=zeros(2*MM+1,2*N+1); c_ds4=zeros(2*MM+1,1);
 

for j = 1 : 2*MM+1    
c_r22(j,1:2*N+1)= c_r(ceil(length(S)/2)-N+(j-MM-1) : ceil(length(S)/2)+N+(j-MM-1));    
c_s4(j)= T/(2*N+1)*c_r(ceil(length(S)/2)-N : ceil(length(S)/2)+N).'*flip(c_r22(j, : )).';
c_ds4(j)=const*1i*(j-MM-1)*c_s4(j);  
end

c_w=zeros(2*L+1,1); c_w_spot=zeros(2*L+1,length(tau)); c_ds5=zeros(2*L+1, 2*M +1);
    

for j = 1 : 2*L+1
c_ds5(j,1:2*M +1)= c_ds4 (  ceil(length(c_ds4)/2) -M  +(j-L-1 ) : ceil(length(c_ds4)/2) +M  +(j-L-1 ) );
c_w(j)=  T   /(2*M+1)*  c_ds.'*flip(c_ds5(j,:)).';
  for ii=1 : length(tau)
  c_w_spot(j,ii)=c_w(j)*(1-abs(j-L-1)/L)*exp(const*tau(ii)*(j-L-1)*1i);
  end
end

VV1_spot=real(sum(c_w_spot));  



%% estimation of logret coeffs 2

 
c_r1b=zeros(length(S1),1); 
tt2=-1i*const*t2(1:end-1);


for j = 1:  length(S1)  
     
c_r1b(j)= exp(S1(j)*tt2).'*rb;  
    
end 
 
c_rb=1/T*[flip(conj(c_r1b));  sum(rb); c_r1b];




%% estimation of vol coeffs 2
 


c_sb=zeros(2*M+1,1);  c_dsb=zeros(2*M+1,1); c_r2b=zeros(2*M+1,2*N+1);
 

for j = 1 : 2*M+1
c_r2b(j,1:2*N+1)=   c_rb(ceil(length(S)/2)-N+(j-M-1) : ceil(length(S)/2)+N+(j-M-1));    
c_sb(j)=T/(2*N+1)*c_rb(ceil(length(S)/2)-N : ceil(length(S)/2)+N).'*flip(c_r2b(j,:)).';
c_dsb(j)=const*1i*(j-M-1 )*c_sb(j);
 
end


 
%% estimation of volvol coeffs 2

 
MM=M+L;
 
c_s4b=zeros(2*MM+1,1); c_r22b=zeros(2*MM+1,2*N+1); c_ds4b=zeros(2*MM+1,1);
 

for j = 1 : 2*MM+1    
c_r22b(j,1:2*N+1)= c_rb(ceil(length(S)/2)-N+(j-MM-1) : ceil(length(S)/2)+N+(j-MM-1));    
c_s4b(j)= T/(2*N+1)*c_rb(ceil(length(S)/2)-N : ceil(length(S)/2)+N).'*flip(c_r22b(j, : )).';
c_ds4b(j)=const*1i*(j-MM-1)*c_s4b(j);  
end

c_wb=zeros(2*L+1,1); c_w_spotb=zeros(2*L+1,length(tau)); c_ds5b=zeros(2*L+1, 2*M +1);
    

for j = 1 : 2*L+1
c_ds5b(j,1:2*M +1)= c_ds4b (  ceil(length(c_ds4b)/2) -M  +(j-L-1 ) : ceil(length(c_ds4b)/2) +M  +(j-L-1 ) );
c_wb(j)=  T   /(2*M+1)*  c_dsb.'*flip(c_ds5b(j,:)).';
  for ii=1 : length(tau)
  c_w_spotb(j,ii)=c_wb(j)*(1-abs(j-L-1)/L)*exp(const*tau(ii)*(j-L-1)*1i);
  end
end

VV2_spot=real(sum(c_w_spotb));  


 
%% estimation of covolvol coeffs  



c_wc=zeros(2*L+1,1); c_w_spotc=zeros(2*L+1,length(tau)); 
    

for j = 1 : 2*L+1
 c_wc(j)=  T   /(2*M+1)*  c_ds.'*flip(c_ds5b (j,:)).';
  for ii=1 : length(tau)
  c_w_spotc(j,ii)=c_wb(j)*(1-abs(j-L-1)/L)*exp(const*tau(ii)*(j-L-1)*1i);
  end
end

CVV_spot=real(sum(c_w_spotc));  

end
