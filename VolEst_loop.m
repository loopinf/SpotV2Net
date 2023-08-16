clear
clc

folder_path = 'rawdata/taq/bycomp';
output_folder_vol = 'processed_data/vol/';
output_folder_covol = 'processed_data/covol/';

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
    X = csvread(fullfile(folder_path, csv_files(w).name),1,1);
    for k = w+1:num_files
        % Load data from the second file
        Y = csvread(fullfile(folder_path, csv_files(k).name),1,1);

        % Check if csv_files(k).name or csv_files(w).name is equal to "GS" or "JPM"
        [~, name_k, ~] = fileparts(csv_files(k).name);
        [~, name_w, ~] = fileparts(csv_files(w).name);
        name_k = strrep(name_k, '_20_23', '');
        name_w = strrep(name_w, '_20_23', '');
        if ~(strcmp(name_k, 'GS') || strcmp(name_k, 'JPM') || strcmp(name_w, 'GS') || strcmp(name_w, 'JPM'))
            continue;
        end
        
        current_iteration = current_iteration + 1;
%         if current_iteration < 416
%             % Update waitbar
%             waitbar(current_iteration / total_iterations, h, sprintf('Processing files... (%d/%d)', current_iteration, total_iterations));
%             continue;
%         end
 
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
         
         
        for e = 1 : S % replace 3 with S
            tic
        [C_spot,V1_spot,V2_spot] = FM_spot_cov_noise_trunc(Px(:,e),Py(:,e),t1,t2,tau,T);
        C_matrix(:,e)= C_spot;
        V1_matrix(:,e)= V1_spot;
        V2_matrix(:,e)= V2_spot;
            toc
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
            csvwrite(output_vol1_filename, V1_matrix);
        end
        
        output_vol2_filename = fullfile(output_folder_vol, [name2 '.csv']);
        if exist(output_vol2_filename, 'file') == 2
            % File already exists. Do nothing.
        else
            % Perform your file-saving operation here
            csvwrite(output_vol2_filename, V2_matrix);
        end
        output_covol_filename = fullfile(output_folder_covol, [name1 '_' name2 '.csv']);
        csvwrite(output_covol_filename, C_matrix);

        % Update waitbar
        waitbar(current_iteration / total_iterations, h, sprintf('Processing files... (%d/%d)', current_iteration, total_iterations));

    end
end
% Close waitbar
close(h);