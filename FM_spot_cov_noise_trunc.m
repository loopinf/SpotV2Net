function [C_spot,V1_spot,V2_spot,tau] = FM_spot_cov_noise_trunc(x1,x2,t1,t2,tau,T)

const=2*pi/T;

% Make sure that x1,x2,t1,t2 and tau are column vectors.
x1=x1(:);
t1=t1(:);
x2=x2(:);
t2=t2(:);
tau=tau(:);

r1=diff(x1);  
r2=diff(x2);  

n1=length(r1); n2=length(r2);
N=floor(5*min(n1,n2)^0.5 );
M=floor((1/3)*N^0.5);

c1=abs(r1)< 0.5*(T/n1)^(1/2);
c2=abs(r2)< 0.5*(T/n2)^(1/2);

r1=r1.*c1;
r2=r2.*c2; 
 
K=N+M;
k=1:1:K; 
tt1=-1i*const*t1(1:end-1)'; tt2=-1i*const*t2(1:end-1)';
        
c_r1a=zeros(K,1); c_r2a=zeros(K,1); 
        
for j = 1:length(k)  
    c_r1a(j)= exp(k(j)*tt1)*r1;  
    c_r2a(j)= exp(k(j)*tt2)*r2;              
end 
                
c_r1=1/T* [ flip(conj(c_r1a))  ; sum(r1)  ; c_r1a]; % Fourier coefficients of dx1
c_r2=1/T* [ flip(conj(c_r2a))  ; sum(r2)  ; c_r2a]; % Fourier coefficients of dx2
              
c_c=zeros(2*M+1,1); c_v1=zeros(2*M+1,1); c_v2=zeros(2*M+1,1); 
c_r_aux1=zeros(2*M+1,2*N+1); c_r_aux2=zeros(2*M+1,2*N+1);
c_v1_spot=zeros(2*M+1,length(tau)); c_v2_spot=zeros(2*M+1,length(tau)); c_c_spot=zeros(2*M+1,length(tau)); 
        
center=K+1;

for j = 1 : 2*M+1        
    
    c_r_aux1(j,1:2*N+1)=c_r1(center-N+(j-M-1):center+N+(j-M-1));
    c_r_aux2(j,1:2*N+1)=c_r2(center-N+(j-M-1):center+N+(j-M-1));
    
    c_v1(j)=T/(2*N+1)*c_r1(center-N:center+N).'*flip(c_r_aux1(j,:)).';
    c_v2(j)=T/(2*N+1)*c_r2(center-N:center+N).'*flip(c_r_aux2(j,:)).';
    c_c(j)=T/(2*N+1)*c_r2(center-N:center+N).'*flip(c_r_aux1(j,:)).';
    
    for ii=1 : length(tau)
          c_v1_spot(j,ii)=c_v1(j)*(1-abs(j-M-1)/M)*exp(const*tau(ii)*(j-M-1)*1i);  % feasible variance process 1
          c_v2_spot(j,ii)=c_v2(j)*(1-abs(j-M-1)/M)*exp(const*tau(ii)*(j-M-1)*1i);  % feasible variance process 2
          c_c_spot(j,ii)=c_c(j)*(1-abs(j-M-1)/M)*exp(const*tau(ii)*(j-M-1)*1i);  % feasible covariance process
    end         
end
               
C_spot=real(sum(c_c_spot)); V1_spot=real(sum(c_v1_spot)); V2_spot=real(sum(c_v2_spot)); 


end
