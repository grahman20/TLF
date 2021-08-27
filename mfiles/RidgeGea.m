function Ps = RidgeGea(Ws, Wt,sigma_ridge)
%RIDGEGEA Summary of this function goes here
    
    %% Data normalization
%     Ws = Ws * diag(sparse(1 ./ sqrt(sum(Ws.^2)))); 
%     Wt = Wt * diag(sparse(1 ./ sqrt(sum(Wt.^2))));     
    Ps=[];    
    [npt,dt]=size(Wt);
    kvalues=sigma_ridge;
    for i=1:dt
        Y=Wt(:,i);
        b=ridge(Y,Ws,kvalues,0);
        Ps=[Ps,b];    
    end
    Ps(Ps<0)=0;
    Ps = Ps * diag(sparse(1 ./ sqrt(sum(Ps.^2))));
end
