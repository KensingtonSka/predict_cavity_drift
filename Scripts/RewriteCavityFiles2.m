clear all; close all; clc;
%Great page: https://au.mathworks.com/help/matlab/matlab_prog/compute-elapsed-time.html

root = cd;
% get the folder contents
d = dir(root);
% remove all files (isdir property is 0)
dfolders = d([d(:).isdir]);
% remove '.' and '..' 
dfolders = dfolders(~ismember({dfolders(:).name},{'.','..'}));
filenames = {'FMCavityDrift', 'AMCavityDrift'};

%Messages:
disp('Current directory: ')
disp(root)
disp(' ')
disp(['Number of folders to loop: ' num2str(length(dfolders))])
disp(['First folder is: ' dfolders(1).name])
disp(['Last folder is: ' dfolders(end).name])
disp(['Folders are accessed in ascending numerical order followed by descending alphabetical order.'])


%Ask user to specify the number of folders to loop:  
prompt = {'Start folder num:','End folder num:'};
dlgtitle = 'Confirm Rewrite:';
dims = [1 35];
definput = {'1',num2str(length(dfolders))};
answer = inputdlg(prompt,dlgtitle,dims,definput);
if answer{1} <= 0; answer{1} = 1; end
if answer{2} > length(dfolders); answer{2} = length(dfolders); end
if isempty(answer); disp('Cancelling script!'); end

if ~isempty(answer)
    disp(' ')
    disp('Recreating all .lvm files to be found!')
    rewrite_counter = 0;
    
    %Some clean up because answer isn't always the same variable 'type'
    bounds = [0 0];
    for ii = 1:2
        if strcmp(class(answer{ii}), 'char')
            bounds(ii) = str2double(answer{ii});
        else%if strcmp(class(answer{1}), 'double')
            bounds(ii) = answer{ii};
        end
    end

    
    %Loop through all folders:
    for ii = bounds(2):-1:bounds(1)
        folder = dfolders(ii).name;

        sub = dir([root '\' folder]);
        subfiles = sub([sub(:).isdir] == 0);
        lvm_files = contains({subfiles(:).name},'.lvm'); 
        new_files = ~contains({subfiles(:).name},'new_'); %Files already edited by this code
        date_files = ~contains({subfiles(:).name},[folder '_']); %Files already edited by this code
        subfiles = subfiles(new_files & date_files & lvm_files);
        
        disp([num2str(length(subfiles)) ' in ' folder])

        if ~isempty(subfiles)
            cd([root '\' folder])
            %Loop through all lvm files:
            for jj = 1:length(subfiles)
                % Get the file name and last file write date:
                file = subfiles(jj).name;
                RECtime = subfiles(jj).date;
                endDate = datetime(RECtime); 

                %Load the data:
                data = load([root '\' folder '\' file]);
                
                %Get creation time and add timepoints:
                startDate = endDate - seconds(data(end,1));
                sampleDates = startDate + seconds(data(:,1));
                
                %Convert to date strings:
                sampleDates = datestr(sampleDates, 'hh:MM:ss.fff');

                %Write ze data to a new file:
                fid = fopen([folder '_' file],'wt');
                for kk = 1:length(:,sampleDates)
                    nbyte = fprintf(fid, '%f\t%f\t%s\n', data(kk,1), data(kk,2), sampleDates(kk,:));
                    
                    %Progress display:
                    clc;
                    disp(['Folder: ' folder ' (' num2str(ii) '/' num2str(bounds(2)) ')'])
                    disp(['File ' num2str(jj) '/' num2str(length(subfiles))])
                    bar = strings(20,1);
                    bar(:) = ' ';
                    bar(1:round(20*kk/length(sampleDates))) = '=';
                    disp(strjoin(['[' strjoin(bar,'') ']' ],''))
                end
                fclose(fid);
                
                rewrite_counter = rewrite_counter + 1;
            end
        end
    end
    disp([num2str(rewrite_counter) ' .lvm files rewritten!'])
end





