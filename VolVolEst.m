clear
clc

%% data loading and preparation

X=csvread('MMM.csv',1,1);  
Y=csvread('HD.csv',1,1);  
 
XX=reshape(X,[],1);
YY=reshape(Y,[],1);

Px=log(XX);
Py=log(YY);

S=min(floor(size(X)/20)); % n. of months
T=20;
n=max(size(X))*T-1;
t1=0:T/n:T;
t2=t1;
tau=0:1/13:T; 

CVV_matrix=zeros(length(tau),S);
VV1_matrix=zeros(length(tau),S); 
VV2_matrix=zeros(length(tau),S); 
 

%% estimation
 
for e = 1 : 1 % replace 1 with S
     
[C_spot,V1_spot,V2_spot] = FM_spot_covolvol_noise_trunc(Px(1+(e-1)*23401*T: e*23401*T) ,Py(1+(e-1)*23401*T: e*23401*T),t1,t2,tau,T);
CVV_matrix(:,e)= C_spot;
VV1_matrix(:,e)= V1_spot;
VV2_matrix(:,e)= V2_spot;
    
end
 
%%  sample plots
u=1;

figure
plot(CVV_matrix(:,u))

 figure
 plot(VV1_matrix(:,u))

 figure
 plot(VV2_matrix(:,u))