clear;clc;
%% set inputs 
basepath='C:\Gea\Research\Experiment\TL';

% datasets = {'COIL','MNIST_USPS'};
datasets = {'Test'};

methods = {'TLF'};

%% process the datasets for each method
nds=size(datasets,2);
nmethods=size(methods,2);
Runs=1;
for ds=1:nds  % loop for each dataset
    for run=1:Runs % loop for each run
        for method=1:nmethods % loop for each method  
            method_name=methods{method};
            path=strcat(basepath,'\',datasets{ds},'\Run-',num2str(run,'%d'),'\',method_name,'\');
            logf=[path 'log.txt'];
            outf=strcat(path,method_name,'_accuracy.csv');

            fprintf('Log file: %s\n\n',logf);
            logfid=fopen(logf,'r');

            outfid=fopen(outf,'w');
            header='Source, Target, Accuracy, ExecutionaTime';
            fprintf(outfid,'%s\n',header);
            fclose(outfid);
            outfid=fopen(outf,'a');
            line=fgetl(logfid);
            while ischar(line)
                fprintf('\nProcessing line: %s\n',line);
                files=split(line,',');
                sf=files{1}; tf=files{2}; testf=files{3};
                srcFile=[path sf]; tgtFile=[path tf]; testFile=[path testf];
                t0 = clock;
                Acc=0.0;
%                 try 
                    if strcmp(method_name,'TLF')                        
                        P=FindProjection(srcFile,tgtFile);
                        disp(P);
%                         [Acc] = RunTLF(srcFile,tgtFile,testFile);                    
                    end
%                 catch ME
%                     disp(ME);
%                 end    
                ms = round(etime(clock,t0) * 1000);
                accuracy= Acc * 100;
                fprintf('%s,%s,%.2f%%,%d\n', sf, tf,accuracy,ms);
                fprintf(outfid,'%s, %s, %.2f%%,%d\n', sf, tf,accuracy,ms);
                line = fgetl(logfid);
            end
            fclose(logfid);
            fclose(outfid);
        end
    end
end