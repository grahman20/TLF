function [X,Y]= ReadInstances(D)
%READINSTANCES Summary of this function goes here
    %## dataset
    numAttr = D.numAttributes;
    numInst = D.numInstances;
    classIndex=numAttr-1;
    D.setClassIndex(classIndex);
    
    %## instances
    X = zeros(numInst,numAttr-1);
    for i=1:numAttr-1
        X(:,i) = D.attributeToDoubleArray(i-1);
    end
    Y=zeros(numInst,1);
    for i=1:numInst
      inst=D.get(i-1);
      Y(i,1)=inst.value(classIndex)+1;
%       Y(i,1) = D.attribute(classIndex).value(v);
    end
end
