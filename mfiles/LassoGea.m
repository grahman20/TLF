function Ps = LassoGea(Ws, Wt)
%LASSOGEA Summary of this function goes here
Ps=[];
[npt,dt]=size(Wt);
for i=1:dt
    Y=Wt(:,i);
    [b,fitinfo] = lasso(Ws,Y);
%     b(b<0)=0;
    [mv, idxLambda1SE]=min(fitinfo.MSE);
    coef = b(:,idxLambda1SE);
    coef0 = fitinfo.Intercept(idxLambda1SE);
    coef=[coef0;coef];
    Ps=[Ps,coef];    
end
end

