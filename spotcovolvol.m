clear
clc

%% Simulation setup 

rng(1989) % seed
T=1; % horizon of each trajectory (in days)
n=23400; % number of prices simulated in one trajectory
dt=T/n; % price simulation frequency 
xi1=0; % noise to signal ratio (put 0 if you do not want to simulate the noise)
xi2=0;
 
mi=0.05; a=0.5; th=3; g=1.2; % Heston parameters (daily)
mi2= 0; a2=0.5; th2=3 ; g2=1.2 ;  
 
xx=0.5;
rho12=xx; rho13=-xx; rho14=0; rho23=0; rho24=-xx; rho34=xx;

mu=zeros(4,1);  
sigma = [1	rho12	rho13	rho14 ; rho12	1	rho23	rho24 ; rho13	rho23	1	rho34; rho14	rho24	rho34	1];
 
 %% Simulation
 
  
P1=zeros(1, n+1);
P2=zeros(1, n+1);
V1=zeros(1, n+1);
V2=zeros(1, n+1);

V1(1)=gamrnd(2*th*a/g^2,g^2/2/th);
V2(1)=gamrnd(2*th2*a2/g2^2,g2^2/2/th2);
P1(1)=log(100);
P2(1)=log(100);


dW = mvnrnd(mu,sigma, n+1);


for nn = 2 :  n+1 
    V1(nn)  = V1(nn-1)  + th*(a-V1(nn-1))*dt          +   g* sqrt(V1(nn-1))*sqrt(dt)*dW(nn-1,1);
    
    V2(nn) = V2(nn-1)  + th2*(a2-V2(nn-1))*dt          +   g2* sqrt(V2(nn-1))*sqrt(dt)*dW(nn-1,2);
    
    P1(nn) = P1(nn-1)  +    (mi-0.5*V1(nn-1))*dt     +      sqrt(V1(nn-1))*sqrt(dt)*dW(nn-1,3);
    
    P2(nn) = P2(nn-1)  +    (mi2-0.5*V1(nn-1))*dt     +      sqrt(V2(nn-1))*sqrt(dt)*dW(nn-1,4);
 
    
end
 


% Adding noise:

std_r=std(diff(real(P1)));
std_r_bis=std(diff(real(P2)));

eta=randn(n+1,1);
eta_bis= randn(n+1,1);

eta2=xi1*std_r*eta;
eta2_bis=xi2*std_r_bis*eta_bis;

x1=P1+eta2'; x2=P2+eta2_bis';
%%

const=2*pi/T; % scaling constant for time
f1=1; % price sampling frequency (in seconds)
f2=1800/f1; % estimation freqeuncy for spot quantities (in units of f1)
P1=downsample(x1,f1); % sparse sampling (if f1=1, uses all the available prices)
P2=downsample(x2,f1); % sparse sampling (if f1=1, uses all the available prices)
r=diff(P1)'; % log-returns
rb=diff(P2)'; % log-returns
n=length(r); N=floor(n/2); M=floor(0.35*N^0.5); L=floor(M^0.5); % frequencies (costants not carefully optimized) 
t=0:T/n:T; t=t'; % price sampling grid
tau=0:f2*T/n:T; % estimation grid
S=-L-M-N:1:N+M+L;

%%  Estimation of log-ret coeffs

% questa routine è stata fatta ottimizzando la velocità di esecuzione

S1=1:1:N+M+L;

tic
 
c_r1=zeros(length(S1),1); 
t1=-1i*const*t(1:end-1);
c_rb1=zeros(length(S1),1); 



for j = 1:  length(S1)  

c_r1(j)= exp(S1(j)*t1).'*r;  
c_rb1(j)= exp(S1(j)*t1).'*rb;   

end 
 
c_r=1/T* [flip(conj(c_r1)); sum(r); c_r1];
c_rb=1/T* [flip(conj(c_rb1)); sum(r); c_rb1];

toc


%% estimation of vol coeffs
 
tic

c_s=zeros(2*M+1,1); c_s_spot=zeros(2*M+1,length(tau)); c_ds=zeros(2*M+1,1); c_r2=zeros(2*M+1,2*N+1);
 

for j = 1 : 2*M+1
c_r2(j,1:2*N+1)=   c_r(ceil(length(S)/2)-N+(j-M-1) : ceil(length(S)/2)+N+(j-M-1));    
c_s(j)=T/(2*N+1)*c_r(ceil(length(S)/2)-N : ceil(length(S)/2)+N).'*flip(c_r2(j,:)).';
c_ds(j)=const*1i*(j-M-1 )*c_s(j);
 
end

toc

tic

c_sb=zeros(2*M+1,1); c_sb_spot=zeros(2*M+1,length(tau)); c_dsb=zeros(2*M+1,1); c_r2b=zeros(2*M+1,2*N+1);
 

for j = 1 : 2*M+1
c_r2b(j,1:2*N+1)=   c_rb(ceil(length(S)/2)-N+(j-M-1) : ceil(length(S)/2)+N+(j-M-1));    
c_sb(j)=T/(2*N+1)*c_rb(ceil(length(S)/2)-N : ceil(length(S)/2)+N).'*flip(c_r2b(j,:)).';
c_dsb(j)=const*1i*(j-M-1 )*c_sb(j);
 
end

toc
 
%% estimation of volvol coeffs

W = rho12*g*g^2*sqrt(V1).*sqrt(V2); % true volvol process
W=g^2*V1;

tic
 
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

W_spot=real(sum(c_w_spot));  

toc



%% check of estimates

W_comp=downsample(W, f1*f2);
  
figure

plot(W_spot)
hold on
plot(W_comp)  
 

  
  
 
  
 