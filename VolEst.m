clear
clc

X= csvread('rawdata/taq/bycomp/IBM_20_23.csv',1,1);  
Y= csvread('rawdata/taq/bycomp/MSFT_20_23.csv',1,1);  
 
Px=log(X);
Py=log(Y);

S=min(size(Px)); % n. of days
n=max(size(Px))-1;

 
T=1;
t1=0:T/n:T;
t2=t1;
tau=0:T/13:T; 

 C_matrix=zeros(length(tau),S);
 V1_matrix=zeros(length(tau),S); 
 V2_matrix=zeros(length(tau),S); 
 
 
for e = 1 : 3 % replace 3 with S
    tic
[C_spot,V1_spot,V2_spot] = FM_spot_cov_noise_trunc(Px(:,e),Py(:,e),t1,t2,tau,T);
C_matrix(:,e)= C_spot;
V1_matrix(:,e)= V1_spot;
V2_matrix(:,e)= V2_spot;
    toc
end
 

% sample plots
u=2;

figure
plot(C_matrix(:,u))

 figure
 plot(V1_matrix(:,u))

 figure
 plot(V2_matrix(:,u))