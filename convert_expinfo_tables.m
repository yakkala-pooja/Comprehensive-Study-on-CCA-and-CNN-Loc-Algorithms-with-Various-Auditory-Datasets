% MATLAB script to convert expinfo tables to structs for Python compatibility
% 
% This script:
% 1. Loads each S*.mat file from Data/Fulsang/EEG/
% 2. Extracts expinfo table
% 3. Converts table to struct using table2struct
% 4. Saves as S*_expinfo_struct.mat in the same directory
%
% Usage:
%   Run this script once in MATLAB:
%   >> convert_expinfo_tables
%
% Then FULPRE.py will automatically use these converted files.

function convert_expinfo_tables()
    % Set path to EEG directory
    eeg_dir = fullfile('Data', 'Fulsang', 'EEG');
    
    if ~exist(eeg_dir, 'dir')
        error('EEG directory not found: %s', eeg_dir);
    end
    
    % Get all S*.mat files
    mat_files = dir(fullfile(eeg_dir, 'S*.mat'));
    
    if isempty(mat_files)
        error('No S*.mat files found in %s', eeg_dir);
    end
    
    fprintf('Found %d MATLAB files to convert\n', length(mat_files));
    fprintf('%s\n', repmat('=', 1, 70));
    
    success_count = 0;
    fail_count = 0;
    
    for i = 1:length(mat_files)
        mat_file = mat_files(i);
        file_path = fullfile(eeg_dir, mat_file.name);
        
        fprintf('\n[%d/%d] Processing %s...\n', i, length(mat_files), mat_file.name);
        
        try
            % Load MATLAB file
            fprintf('  Loading file...\n');
            loaded = load(file_path);
            
            % Check for expinfo
            if ~isfield(loaded, 'expinfo')
                fprintf('  [WARNING] expinfo not found in %s, skipping\n', mat_file.name);
                fail_count = fail_count + 1;
                continue;
            end
            
            expinfo = loaded.expinfo;
            
            % Check if it's a table
            if ~istable(expinfo)
                fprintf('  [INFO] expinfo is not a table (type: %s), converting anyway...\n', class(expinfo));
            else
                fprintf('  [OK] expinfo is a table, converting to struct...\n');
            end
            
            % Convert table to struct
            % Use 'ToScalar', true to create scalar struct with array fields
            if istable(expinfo)
                expinfo_struct = table2struct(expinfo, 'ToScalar', true);
            else
                % If it's already a struct, use it directly
                expinfo_struct = expinfo;
            end
            
            % Create output filename
            [~, base_name, ~] = fileparts(mat_file.name);
            output_file = fullfile(eeg_dir, [base_name, '_expinfo_struct.mat']);
            
            % Save converted struct
            fprintf('  Saving to %s...\n', [base_name, '_expinfo_struct.mat']);
            save(output_file, 'expinfo_struct', '-v7');
            
            % Verify attend_mf exists
            if isfield(expinfo_struct, 'attend_mf')
                attend_mf = expinfo_struct.attend_mf;
                fprintf('  [SUCCESS] attend_mf found: %d trials\n', length(attend_mf));
                fprintf('  attend_mf values: %s\n', mat2str(unique(attend_mf)));
            else
                fprintf('  [WARNING] attend_mf not found in converted struct\n');
                fprintf('  Available fields: %s\n', strjoin(fieldnames(expinfo_struct), ', '));
            end
            
            success_count = success_count + 1;
            
        catch ME
            fprintf('  [ERROR] Failed to process %s: %s\n', mat_file.name, ME.message);
            fail_count = fail_count + 1;
        end
    end
    
    fprintf('\n');
    fprintf('%s\n', repmat('=', 1, 70));
    fprintf('Conversion complete!\n');
    fprintf('  Success: %d files\n', success_count);
    fprintf('  Failed:  %d files\n', fail_count);
    fprintf('\n');
    fprintf('FULPRE.py will now automatically use these converted files.\n');
    fprintf('Run your preprocessing again.\n');
end
