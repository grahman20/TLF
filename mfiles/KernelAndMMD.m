function [P] = KernelAndMMD(SrcArff,tgtArff,L,sigma_ridge, lambda_mmd, gamma_manifold)
%KERNELANDMMD Summary of this function goes here
%   Detailed explanation goes here
    %
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
    options.rho = gamma_manifold;%1.0;
    options.p = 10;
    options.lambda =lambda_mmd;%10.0;
    options.eta =sigma_ridge;% 0.1;
    options.T = 1;
    
    %% Build classifier for the target domain
    P = kernel_mmd(Xs,Ys,Xt,Yt,L,options);      
end


function P = kernel_mmd(Xs,Ys,Xt,Yt,L,options)

% Reference:
%% Jindong Wang, Wenjie Feng, Yiqiang Chen, Han Yu, Meiyu Huang, Philip S.
%% Yu. Visual Domain Adaptation with Manifold Embedded Distribution
%% Alignment. ACM Multimedia conference 2018.

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% K      :  kernel matrix
%%%% M      :  MMD matrix
%%%% L     :  graph laplacian matrix
%%%% P     :  Cofficient matrix
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end

    % Manifold feature learning
    [Xs_new,Xt_new,~] = GFK_Map(Xs,Xt,options.d);
    Xs = double(Xs_new');
%     disp(Xs');
    
    Xt = double(Xt_new');
%     disp(Xt');
    
    X = [Xs,Xt];
    n = size(Xs,2);
    m = size(Xt,2);   


    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));

    %% Construct graph Laplacian
%     if options.rho > 0
%         manifold.k = options.p;
%         manifold.Metric = 'Cosine';
%         manifold.NeighborMode = 'KNN';
%         manifold.WeightMode = 'Cosine';
%         W = lapgraph(X',manifold);
%         Dw = diag(sparse(sqrt(1 ./ sum(W))));
%         L = eye(n + m) - Dw * W * Dw;
%     else
%         L = 0;
%     end


    %% Construct kernel and labeled matrix
    K = rbfKernel('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));   
    E = diag(sparse([ones(n,1);ones(m,1)]));
    %% construct MMD
    if options.lambda>0
        % Estimate mu
        mu = estimate_mu(Xs',Ys,Xt',Yt);
        % Construct MMD matrix
        e = [1 / n * ones(n,1); -1 / m * ones(m,1)];
        M = e * e' * length(unique(Ys));
        N = 0;
        for c = reshape(unique(Ys),1,length(unique(Ys)))
            e = ones(n + m,1);
            e(Ys == c) = 1 / length(find(Ys == c));
            e(n + find(Yt == c)) = -1 / length(find(Yt == c));
            e(isinf(e)) = 0;
            N = N + e * e';
        end
        M = (1 - mu) * M + mu * N;
        M = M / norm(M,'fro');
    else
        M =0;
    end
    %% Compute coefficients vector Beta
    if options.lambda>0 && options.rho>0
%         P = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E*X');
%         P = (options.lambda * M + options.rho * L) * K;% + options.eta * speye(n + m,n + m);  
        P = ((E + options.lambda * M + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E);
    elseif options.lambda>0 
       P = ((E + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E);     
    elseif options.rho>0 
       P = ((E + options.lambda * M ) * K + options.eta * speye(n + m,n + m)) \ (E);
    else 
       P = (E * K + options.eta * speye(n + m,n + m)) \ E;   
    end
%         disp(P);
%         ds=size(Xs,1);
%         dt=size(Xt,1);
%         P=P(1:ds,1:dt);
        P=X*P*X';
        P(P<0)=0;
%         disp(P);
        P = P * diag(sparse(1 ./ sqrt(sum(P.^2))));
%         g1=G(1:n,1:n);
%         g2=G(n+1:n+m,1:n);
%         g3=G(1:n,n+1:n+m);
%         g4=G(n+1:n+m,n+1:n+m);
%         
%         Gnew=g1*g3*g4;       
%         P=Xs*Gnew*Xt';
%         P=max(P,0);
        

end

function K = rbfKernel(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end
