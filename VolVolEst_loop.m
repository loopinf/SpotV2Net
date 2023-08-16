clear
clc

folder_path = 'rawdata/taq/bycomp';
output_folder_vol = 'processed_data/vol_of_vol/';
output_folder_covol = 'processed_data/covol_of_vol/';

% Get list of all .mat files in the folder
csv_files = dir(fullfile(folder_path, '*.csv'));

num_files = length(csv_files);

% Create a waitbar
h = waitbar(0, 'Computing Variance-Covariance Pairs...');

total_iterations = num_files * (num_files - 1) / 2;
current_iteration = 0;

%%
for w = 1:num_files
    % Load data from the first file
    X1 = csvread(fullfile(folder_path, csv_files(w).name),1,1);
    for k = w+1:num_files
        % Load data from the second file
        Y1 = csvread(fullfile(folder_path, csv_files(k).name),1,1);
        
        current_iteration = current_iteration + 1;

        X=downsample(X1,5);  
        Y=downsample(Y1,5);  

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
         
        for e = 1 : S % replace 1 with S
             
        %[C_spot,V1_spot,V2_spot] = FM_spot_covolvol_noise_trunc(Px(1+(e-1)*23401*T: e*23401*T) ,Py(1+(e-1)*23401*T: e*23401*T),t1,t2,tau,T);
        [C_spot,V1_spot,V2_spot] = FM_spot_covolvol_noise_trunc(Px(1+(e-1)*4681*T: e*4681*T) ,Py(1+(e-1)*4681*T: e*4681*T),t1,t2,tau,T); % new
        CVV_matrix(:,e)= C_spot;
        VV1_matrix(:,e)= V1_spot;
        VV2_matrix(:,e)= V2_spot;
            
        end

        % Extract the names from the filenames
        [~, name1, ~] = fileparts(csv_files(w).name);
        [~, name2, ~] = fileparts(csv_files(k).name);
        
        % Remove suffix from the names
        name1 = strrep(name1, '_20_23', '');
        name2 = strrep(name2, '_20_23', '');

        disp([name1 ' ' num2str(w) ', ' name2 ' ' num2str(k)])
        
        % Save matrices to a new file in the output folder
        output_vol1_filename = fullfile(output_folder_vol, [name1 '.csv']);
        if exist(output_vol1_filename, 'file') == 2
            % File already exists. Do nothing.
        else
            % Perform your file-saving operation here
            csvwrite(output_vol1_filename, VV1_matrix);
        end
        
        output_vol2_filename = fullfile(output_folder_vol, [name2 '.csv']);
        if exist(output_vol2_filename, 'file') == 2
            % File already exists. Do nothing.
        else
            % Perform your file-saving operation here
            csvwrite(output_vol2_filename, VV2_matrix);
        end
        output_covol_filename = fullfile(output_folder_covol, [name1 '_' name2 '.csv']);
        csvwrite(output_covol_filename, CVV_matrix);

        % Update waitbar
        waitbar(current_iteration / total_iterations, h, sprintf('Processing files... (%d/%d)', current_iteration, total_iterations));

    end
end
% Close waitbar
close(h);