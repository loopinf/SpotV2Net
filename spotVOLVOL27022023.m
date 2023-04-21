clear
clc
close all

%% Simulation setup (Heston)

rng(6834) % seed
T=1; % horizon of each trajectory (in trading days; each trading day is 6.5 hours long)
obs=23400*T; % number of 1-second returns simulated in one trajectory
dt=T/obs; % price simulation frequency (in days)
rho=-0.7; % correlation between the two Brownian motions
mi=0.05; a=0.5; th=3; g=1.2; % Heston parameters (daily)
mu=[0;0]; sigma=[1 rho; rho 1];  % Brownian Motion parameters

if 2*a*th-g^2<0 
   error('Error. Feller condition not satisfied')
end

%%  Simulation (Heston)

P=zeros(1, obs+1); % log-prices
V=zeros(1, obs+1); % spot variances
 
V(1)=gamrnd(2*th*a/g^2,g^2/2/th); % extracts the initial variance from the stationary distribution of the CIR
P(1)=log(100);

dW = sqrt(dt)*mvnrnd(mu,sigma, obs+1); % simulates the two correlated Brownian motions

for nn = 2 :  obs+1 
    V(nn)  = V(nn-1) + th*(a-V(nn-1))*dt      + g*sqrt(V(nn-1))*dW(nn-1,1);
    P(nn) = P(nn-1)  +    (mi-0.5*V(nn-1))*dt +   sqrt(V(nn-1))*dW(nn-1,2);
end
 
%%  Estimation setup (Fourier)
  
const=2*pi/T; % scaling constant for time
f1=1; % price sampling frequency (in seconds)
f2=900/f1; % estimation freqeuncy for spot quantities (in units of f1)
P1=downsample(P,f1); % sparse sampling (if f1=1, uses all the available prices)
r=diff(P1)'; % log-returns
n=length(r); N=floor(n/2); M=floor(0.6*N^0.5); L=floor(M^0.5); % frequencies (costants not carefully optimized) 
t=0:T/n:T; t=t'; % price sampling grid
tau=0:f2*T/n:T; % estimation grid
S=-L-M-N:1:N+M+L;

%%  Estimation of log-ret coeffs

% questa routine è stata fatta ottimizzando la velocità di esecuzione

S1=1:1:N+M+L;

tic
 
c_r1=zeros(length(S1),1); 
t1=-1i*const*t(1:end-1);

for j = 1:  length(S1)  
     
c_r1(j)= exp(S1(j)*t1).'*r;  
    
end 
 
c_r=1/T* [flip(conj(c_r1)); sum(r); c_r1];

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
 
%% estimation of volvol coeffs

W = g^2*V; % true volvol process

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
 

  
  
 
  
 