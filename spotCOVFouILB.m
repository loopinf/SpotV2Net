clear
clc

folder_path = 'processed_data/price_tables/';
output_folder = 'processed_data/cov_estimation/';

% Get list of all .mat files in the folder
mat_files = dir(fullfile(folder_path, '*.mat'));

num_files = length(mat_files);

% Create a waitbar
h = waitbar(0, 'Computing Covariance Pairs...');

total_iterations = num_files * (num_files - 1) / 2;
current_iteration = 0;

%% 

for w = 1:num_files
    % Load data from the first file
    load(fullfile(folder_path, mat_files(w).name));
    Pnum1 = Pnum;
    clear Pnum;
    for k = w+1:num_files
        % Load data from the second file
        load(fullfile(folder_path, mat_files(k).name));
        Pnum2 = Pnum;
        clear Pnum;
        
        % Body of the script that calculates C_final
        %%  Estimation setup
         T=1 ;
         zx = 391;
         Y=length(Pnum1)/zx; %-1 ; % n of days  
         C_spot=zeros(14,Y); 
         % for loop day by day, estimation daily vol every 30 mins
        
         for y=1:Y
           tic  
           P1=log(Pnum1(zx*(y-1)+1  :zx*y))';  
           P2=log(Pnum2(zx*(y-1)+1  :zx*y))';   
        
        const=2*pi/T;
        r1=diff(P1)';  
        r2=diff(P2)';  
        
        n=length(r1); 
        N=floor(n/2);
        M=floor(N^0.5);   
           
        t=0:T/n:T; tau=0:30*T/n:T; 
        S= -M-N:1:N+M ;
        
        % Estimation of logret coeffs
         
        S1=1:1:N+M ;
        t1=-1i*const*t(1:end-1);
        
        c_r1a=zeros(length(S1),1); 
        c_r2a=zeros(length(S1),1); 
        
        for j = 1:  length(S1)  
             
        c_r1a(j)= exp(S1(j)*t1.').'*r1;  
        c_r2a(j)= exp(S1(j)*t1.').'*r2;  
            
        end 
         
         
        c_r1=1/T* [ flip(conj(c_r1a))  ; 1/T* sum(r1)  ; c_r1a];
        c_r2=1/T* [ flip(conj(c_r2a))  ; 1/T* sum(r2)  ; c_r2a];
        
         
        c_c=zeros(2*M+1,1); c_c_spot=zeros(2*M+1,length(tau)); c_r_aux=zeros(2*M+1,2*N+1);
         
        
        for j = 1 : 2*M+1
        c_r_aux(j,1:2*N+1)=   c_r1  (  ceil(length(S)/2) -N +( j-M-1 ) : ceil(length(S)/2) +N +( j-M-1 ) ) ;    
        c_c(j)= T/(2*N+1) * c_r2  (  ceil(length(S)/2) -N   : ceil(length(S)/2) +N   ).' * flip(c_r_aux(j, : )).';
        
         
           for ii=1 : length(tau)
          c_c_spot(j,ii)=c_c(j)*(1-abs(j-M-1)/M)*exp(const*tau(ii)*(j-M-1)*1i);  % feasible c_A
           end
           
        end
        
         
        
         
        C_spot(:,y)=real(sum(c_c_spot)); 
        
         end
         
         L=size(C_spot);
         C_final=reshape(C_spot,[L(1)*L(2),1]); 

        % Extract the names from the filenames
        [~, name1, ~] = fileparts(mat_files(w).name);
        [~, name2, ~] = fileparts(mat_files(k).name);
        
        % Remove 'Pnum_' from the names
        name1 = strrep(name1, 'Pnum_', '');
        name2 = strrep(name2, 'Pnum_', '');

        disp([name1 ' ' num2str(w) ', ' name2 ' ' num2str(k)])
        
        % Save C_final to a new file in the output folder
        output_filename = fullfile(output_folder, [name1 '_' name2 '.csv']);
        csvwrite(output_filename, C_final);

        % Update waitbar
        current_iteration = current_iteration + 1;
        waitbar(current_iteration / total_iterations, h, sprintf('Processing files... (%d/%d)', current_iteration, total_iterations));

    end
end
% Close waitbar
close(h);
