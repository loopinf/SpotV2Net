clear
clc

%wbar = waitbar(0, 'Creating Price Tables for Covariances...');
load processed_data/bdates.mat
bdates= filtered_timetable;
bdates = datetime(bdates.Time);

folder = 'rawdata/dji_1min_rthjdp';
files = dir(fullfile(folder, '*.txt'));
%for i = 1:length(files)
filename = files(1).name;
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
 
 [filepath, name, ext] = fileparts(filename);
 name_parts = split(name, '_');
 name = name_parts{1};
 save(fullfile(sprintf('processed_data/price_tables/Pnum_%s.mat', name)), 'Pnum');
 % Update the waitbar with the current progress
 %waitbar(i/numel(files), wbar, sprintf('Iteration %d/%d completed', i, numel(files)));
%end

% close waitbar
close(wbar)



