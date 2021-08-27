function [accuracy] = RunTLF(sourceFile,targetFile, testFile)
%RUNTLF Summary of this function goes here
%This function calculate accuracy for the TLF technique   
    addpath('./TLF/'); %add path of the TLF method
    
    %% Read source file
    [srcX,srcY]=ReadArff(sourceFile); 
    srcX = srcX ./ repmat(sum(srcX,2),1,size(srcX,2)); 
    Xs = zscore(srcX,1);    clear srcX
    Ys=srcY; clear srcY
    
    %% Read target file
    [tgtX,tgtY]=ReadArff(targetFile); 
    tgtX = tgtX ./ repmat(sum(tgtX,2),1,size(tgtX,2)); 
    Xt = zscore(tgtX,1);    clear tgtX
    Yt=tgtY; clear tgtY
    
    %% Set optiones for TLF
    dd=20;
    [numRows,numCols] = size(Xs);
    if dd>numRows
        dd=numRows-2;
    end   
    if dd>numCols
        dd=numCols-2;
    end
    [numRows,numCols] = size(Xt);
    if dd>numRows
        dd=numRows-2;
    end   
    if dd>numCols
        dd=numCols-2;
    end
    options.d = dd; %defualt 20
    options.rho = 1.0;
    options.p = 10;
    options.lambda = 10.0;
    options.eta = 0.1;
    options.T = 1;
    
    %% Build classifier for the target domain
    [accuracy,~,~,Yt_new] = TLF(Xs,Ys,Xt,Yt,options); 
    
    %% Read test file
    [testX,testY]=ReadArff(testFile);  
    testX = testX ./ repmat(sum(testX,2),1,size(testX,2)); 
    Xtest = zscore(testX,1);    clear testX
    Ytest=testY; clear testY
    
    %% Set optiones of TLF for the test data
    dd=20;
    [numRows,numCols] = size(Xtest);
    if dd>numRows
        dd=numRows-2;
    end   
    if dd>numCols
        dd=numCols-2;
    end
    options.d = dd; %defualt 20
    options.T = 1;
    X=[Xs;Xt];
    Y=[Ys;Yt];
    
    %% Calculate accuracy for the test data
    [accuracy,~,~,~] = TLF(X,Y,Xtest,Ytest,options);
end

