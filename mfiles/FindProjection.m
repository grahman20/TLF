function P = FindProjection(SrcArff,tgtArff)
%FINDPROJECTION Summary of this function goes here
%This function calculate accuracy for the TLF technique   
   
    %% Read source file
    [srcX,srcY]=ReadArff(SrcArff); 
    srcX = srcX ./ repmat(sum(srcX,2),1,size(srcX,2)); 
    Xs = zscore(srcX,1);    clear srcX
    Ys=srcY; clear srcY
    
    %% Read target file
    [tgtX,tgtY]=ReadArff(tgtArff); 
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
    options.T = 10;
    
    %% Build classifier for the target domain
    P = TLF(Xs,Ys,Xt,Yt,options);      
end


