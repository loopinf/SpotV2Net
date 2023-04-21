clear
clc

wbar = waitbar(0, 'Estimating Vol...');
load bdates.mat
bdates= filtered_timetable;
bdates = datetime(bdates.Time);

folder = 'rawdata/dji_1min_rthjdp';
files = dir(fullfile(folder, '*.txt'));
for i = 1:length(files)
    filename = files(i).name;
    opts = detectImportOptions(fullfile(folder, filename),'Delimiter',{',' }); 
    opts.SelectedVariableNames = {'Var1','Var2','Var3','Var4','Var5','Var6'  };
    price = readtable(fullfile(folder, filename),opts);
    
    
     %%
    H1 = hour(price.Var1);
    price(H1<9,:) = [];
    
    H2 = hour(price.Var1);
    price(H2>=16 ,:) = [];
      
      %%
     
    
    H4 = hour(price.Var1);
    M4 = minute(price.Var1);
    price((H4 ==9 & M4< 30),:) = [];

    H41 = hour(price.Var1);
    M41 = minute(price.Var1);
    price((H41 ==16 & M41> 0),:) = [];
     
    %%
     
    priceb=table2timetable(price);
    pricec = retime(priceb,   bdates, 'nearest'  ) ;
     
    clear price
    price=timetable2table(pricec);

     %%
    
    D = day(price.Var1);
    Mo=month(price.Var1);
    price((Mo==7 & D ==3),:) = [];
    clear D Mo  
     
    D = day(price.Var1);
    Mo=month(price.Var1);
    price((Mo==12 & D ==24),:) = [];
    clear D Mo  
     %%
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price(( Ye ==2005 &  Mo==11  & D ==25 ),:) = [];
    clear D Mo Ye 
     
     D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price(( Ye ==2006 &  Mo==11  & D ==24 ),:) = [];
    clear D Mo Ye 
     
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price(( Ye ==2007 &  Mo==11  & D ==23),:) = [];
    clear D Mo Ye  
    
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((  Ye ==2008 & Mo==11  & D ==28 )  , :) = [];
    clear D Mo Ye  
     
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((  Ye ==2009 & Mo==11  & D ==27 )  , :) = [];
    clear D Mo Ye  
    
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((  Ye ==2010 & Mo==11  & D ==26 )  , :) = [];
    clear D Mo Ye  
       
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((  Ye ==2011 & Mo==11  & D ==25 )  , :) = [];
    clear D Mo Ye  
        
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((  Ye ==2012 & Mo==11  & D ==23 )  , :) = [];
    clear D Mo Ye  
       
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((  Ye ==2013 & Mo==11  & D ==29 )  , :) = [];
    clear D Mo Ye  
    
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2014 &  Mo==11  & D ==28 )  , :) = [];
    clear D Mo Ye  
    
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2015 & Mo==11  & D ==27 )  , :) = [];
    clear D Mo Ye  
     
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2016 & Mo==11  & D ==25 )  , :) = [];
    clear D Mo Ye  
     
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2017 & Mo==11  & D ==24 )  , :) = [];
    clear D Mo Ye  
       
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2018 & Mo==11  & D ==23 )  , :) = [];
    clear D Mo Ye  
       
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2019 & Mo==11  & D ==29 )  , :) = [];
    clear D Mo Ye  
       
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2020 & Mo==11  & D ==27 )  , :) = [];
    clear D Mo Ye  
     
    D = day(price.Var1);
    Mo=month(price.Var1);
    Ye=year(price.Var1);
    price((   Ye ==2021 & Mo==11  & D ==26  )  , :) = [];
    clear D Mo Ye  
      
      
     
     
     Pnum=table2array(price(:,5));
     
      
     
     
     
     %%  Estimation setup
     T=1 ;
     zx = 391;
     Y=length(Pnum)/zx; %-1 ; % n of days  
     V_spot=zeros(14,Y); % provare con Y-1 per non buttare l'ultimo giorno
     % for loop day by day, estimation daily vol every 30 mins
    
     for y=1:Y
       tic  
       P=log(Pnum(zx*(y-1)+1  :zx*y))';  
        
    const=2*pi/T;
    r=diff(P)';  
    n=length(r); 
    N=floor(0.9*n/2);
    M=floor(N^0.5);   
       % t is frequency (min) of prices and tau of the vols
    t=0:T/n:T;    tau=0:30*T/n:T; 
    S= -M-N:1:N+M ;
    
    %   Estimation of logret coeffs
     
     S1=1:1:N+M ;
     t1=-1i*const*t(1:end-1);
    
     
    c_r1=zeros(length(S1),1); 
    
    for j = 1:  length(S1)  
         
    c_r1(j)= exp(S1(j)*t1.').'*r;  
        
    end 
     
     
    c_r=1/T* [ flip(conj(c_r1))  ; 1/T* sum(r)  ; c_r1];
    
     
    c_s=zeros(2*M+1,1); c_s_spot=zeros(2*M+1,length(tau)); c_r2=zeros(2*M+1,2*N+1);
     
    
    for j = 1 : 2*M+1
    c_r2(j,1:2*N+1)=   c_r (  ceil(length(S)/2) -N +( j-M-1 ) : ceil(length(S)/2) +N +( j-M-1 ) ) ;    
    c_s(j)= T/(2*N+1) * c_r (  ceil(length(S)/2) -N   : ceil(length(S)/2) +N   ).' * flip(c_r2(j, : )).';
    
     
       for ii=1 : length(tau)
      c_s_spot(j,ii)=c_s(j)*(1-abs(j-M-1)/M)*exp(const*tau(ii)*(j-M-1)*1i);  % feasible c_A
       end
       
    end
    
     
    
     
    V_spot(:,y)=real(sum(c_s_spot)); 
     
     
     
     end
     
     L=size(V_spot);
     V_final=reshape(V_spot,[L(1)*L(2),1]); 
     
     %figure
     %plot(V_final)
     [filepath, name, ext] = fileparts(filename);
     filename_to_save = sprintf('processed_data/vol_estimation/%s_1min_vol.txt', name);
     dlmwrite(filename_to_save, V_final, 'delimiter', ' ');
     % Update the waitbar with the current progress
     waitbar(i/numel(files), wbar, sprintf('Iteration %d/%d completed', i, numel(files)));
end

% close waitbar
close(wbar)



