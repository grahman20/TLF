/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/*
 *   RulesToForest.java  
 *   A Framework for Supervised Heterogeneous Transfer Learning 
 *   using Dynamic Distribution Adaptation and Manifold Regularization
 *
 *   @author Md Geaur Rahman and Md Zahidul Islam
 *   School of Computing, Mathematics & Engineering
 *   Charles Sturt University, Bathurst, NSW, Australia
 */
package transferlearning;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author Md Geaur Rahman <gea.bau.edu.bd>
 */
public class RulesToForest {
        /**
         * For serialization.
         */
        private static final long serialVersionUID = -7891225050957072995L;
        private List<Tree> trees;
        private Instances dataset;
        public RulesToForest(String model, Instances dataset)
        {
            trees=new ArrayList<Tree>();
            this.dataset=dataset;
            if(model.equals(""))
            {
                String majorityCV=TLUtils.majorityClassValue(dataset);
                String []tmpConditions=new String[1];
                tmpConditions[0]=":"+majorityCV;  
                Tree tree=new Tree();
                tree.buildInitialTree(tmpConditions);                    
                trees.add(tree);
                
            } 
            else{
                constructRuleToForest(model);   
            }
        }

        public void constructRuleToForest(String treeStr)
        {

           String []rules=treeStr.split("\n");
           if(rules.length>0)           
           {
               int numTree=TLUtils.countTrees(rules);
               int []Loc=new int[numTree];
               String []conditions=TLUtils.processTrees(rules, Loc);
               int totalCons=conditions.length;
               for(int tc=0;tc<numTree;tc++)
               {
                    int ln;
                    if(tc==numTree-1)
                    {
                         ln=totalCons-Loc[tc];
                    }
                    else
                    {
                         ln=Loc[tc+1]-Loc[tc];
                    }
                    int tln=ln+Loc[tc];

                    String []tmpConditions=new String[ln];
                    for(int r=Loc[tc], i=0;r<tln;r++,i++)
                    {
                        tmpConditions[i]=conditions[r];
                    }
                    
                    Tree tree=new Tree();
                    tree.buildInitialTree(tmpConditions);                    
                    trees.add(tree);
                 }
           }
        }
        public List<Tree> getForest()
        {
            return this.trees;
        }
        public int getForestSize()
        {
            return this.trees.size();
        }
        public double classifyInstance(Instance instance)
        {
            double pred=0;
            int []classValues=new int[dataset.numClasses()];
            Arrays.fill(classValues, 0);
            for(Tree tree:trees)
            {
                Node leaf=tree.findLeafForInstance(instance);
                String cv=leaf.getLeafPrediction();
                double cvi=dataset.attribute(dataset.classIndex()).indexOfValue(cv);
                if (cvi>=0 && cvi<dataset.numClasses())
                {
                    classValues[(int)cvi]++;
                }
            }
            int mx=0;
            for(int i=0;i<dataset.numClasses();i++)
            {
                if(classValues[i]>classValues[mx])
                {
                    mx=i;
                }
            }
            pred=mx;
            return pred;
        }
        public int getTotalLeafCount()
        {
            int count=0;
            for(Tree tree:trees)
            {
                count+=tree.getTotalLeafCount();
            }
            return count;
        }
        public List<Node> getTotalLeaves()
        {
            List<Node> leaf=new ArrayList<>();
            for(Tree tree:trees)
            {
                leaf.addAll(tree.getLeaves());
            }
            return leaf;
        }
        public void assignRecords()
        {
            for(Tree tree:trees)
            {
                tree.assignInstancesToLeaf();
            }
        }
        
        @Override
        public String toString()
        {
            String out="";
            int t=0;
            for(Tree tree:trees)
            {
                t++;
                out+="\nTree: "+t+", Total nodes:"+tree.getTotalNodeCount()
                        +", Total leaves:"+tree.getTotalLeafCount()
                        +", Tree depth:"+tree.getTreeDepth()+"\n";
                out+=tree.toString()+"\n";                
            }
            return out;
        }
        
    
    
    
    
    public class Tree{
        final String levelPadding="|   ";
        private Node root;  
        private int totalNodeCount;
        private int totalLeafCount;
        private boolean leafFoundFlag=false;
        private Node findLeaf;
        private int treeDepth;
        private List<Node> leafCollection = new ArrayList<>();
        public Tree()
        {
            root=new Node(null,0,false,0);
            totalNodeCount=0;
            treeDepth=0;
            totalLeafCount=0;
        }

        public int getTreeDepth()
        {
            return treeDepth;
        }
        public List<Node> getLeaves()
        {
            return leafCollection;
        }
        public void buildInitialTree(String []conditions)
        {
             constructTree(conditions);            
        }
        public void constructTreeSingle(String []conditions)//tree with just a single leaf
        {
                String majorityCV="";
                if(conditions[0].contains(":")&& conditions[0].contains("("))
                {
                    majorityCV=conditions[0].substring(conditions[0].indexOf(":")+1, conditions[0].indexOf("("));
                }
                else if(conditions[0].contains(":")&& !conditions[0].contains("("))
                {
                    majorityCV=conditions[0].substring(conditions[0].indexOf(":")+1, conditions[0].length());
                }
                else if(!conditions[0].contains(":")&& conditions[0].contains("("))
                {
                    majorityCV=conditions[0].substring(0, conditions[0].indexOf("("));
                }
                else
                {
                    majorityCV=conditions[0];
                }
             root=new Node(null,0,true,0);                         
             totalLeafCount++;totalNodeCount++;             
             root.setLeafPrediction(majorityCV.trim());
             leafCollection.add(root);
        }
        public void constructTree(String []conditions)
        {
            int n=conditions.length;  
            if(conditions.length==1)
            {
                constructTreeSingle(conditions);
            }
            else{
            Node currentNode=root;
            int nodeIndex=0;
            boolean rootSet=false;
            ArrayList<String> lAttr=new ArrayList<String>();
            ArrayList<String> sVal=new ArrayList<String>();
            ArrayList<String> sOp=new ArrayList<String>();
                      
            for(int c=0;c<n;c++) 
             { 
                 String []cac=TLUtils.separateConditionClass(conditions[c]);
                 String []strVal=TLUtils.convertStringToArray(cac[0]);
                 int nstr=strVal.length;
                 int cl=0,tl=0;
                 //find levels
                 for (int x=0;x<nstr;x++)
                 {
                     if(strVal[x].equals("|"))
                     {
                         cl++;
                     }
                 }
                 //delete old node information 
                 tl=lAttr.size();
                 if(tl>cl)
                 {
                     while(tl>cl)
                     {
                         tl--;
                         lAttr.remove(tl);
                         sVal.remove(tl);
                         sOp.remove(tl);
                         if(currentNode.getParent()!=null)
                         {
                             currentNode=currentNode.getParent();
                         }
                     }
                 }
                 //add new node information
                 if((cl+2)<nstr)
                 {
                     lAttr.add(strVal[cl]);
                     sOp.add(strVal[cl+1]);
                     sVal.add(strVal[cl+2]);
                 }
                 tl=lAttr.size();
                 if(tl==1)
                 {
                    if(!rootSet) { 
                        
                       currentNode.setNodeInfo(TLUtils.findAttrType(lAttr.get(0),dataset), lAttr.get(0));
                       rootSet=true;
                    }
                 }
                 else if(tl>1){
                     int depth=currentNode.getTreeDepth();
                     if(depth<tl-1)
                     {
                         treeDepth=depth+1;
                        Node child=new Node(currentNode,treeDepth,false,++nodeIndex); 
                        int tmp=tl-1;
                        child.setNodeInfo(TLUtils.findAttrType(lAttr.get(tmp),dataset), lAttr.get(tmp));                    
                        child.setSplitInfo(sOp.get(tmp-1), sVal.get(tmp-1));
                        currentNode.addChild(child);
                        currentNode=currentNode.getLastChild();
                     }
                 }
                 if(!cac[1].isEmpty())
                 {
                     tl=lAttr.size();
                     if(tl>0)
                     {
                         treeDepth=currentNode.getTreeDepth()+1;
                         Node child=new Node(currentNode,treeDepth,true,++nodeIndex);                         
                         totalLeafCount++;
                         tl--;
                         child.setSplitInfo(sOp.get(tl), sVal.get(tl));
                         child.setLeafPrediction(cac[1]);
                         child.setTestAttributes(lAttr);
                         currentNode.addChild(child);
                         leafCollection.add(child);
                         lAttr.remove(tl);
                         sVal.remove(tl);
                         sOp.remove(tl);
                     }
                 }                  
            }
            totalNodeCount=nodeIndex+1;
            }
        }

        public void updateTreeDepth()
        {
            this.totalNodeCount=0;
            leafCollection.removeAll(leafCollection);
            totalLeafCount=0;
            treeDepth=0;
            updateTreeDepthAndIndex(root,0);            
        }
        
        public int[]getLeafIndex()
        {
            int []leafIndex=new int[totalLeafCount];
            int i=0;
            for(Node leaf:leafCollection)
            {
                leafIndex[i]=leaf.getNodeIndex();
                i++;
            }
            return leafIndex.clone();
        }
        
        public void displayLeafInfo()
        {
            for(Node leaf:leafCollection)
            {
                System.out.println(leaf.toString());
            }
        }
        public void updateTreeDepthAndIndex(Node node, int depth)
        {           
                node.setNodeIndex(this.totalNodeCount++);
                node.setTreeDepth(depth);
                if(node.isLeaf()){
                    leafCollection.add(node);
                    this.totalLeafCount++;
                    if(depth>treeDepth)treeDepth=depth;
                }
                List<Node> children=node.getChildren();
                for (Node child : children) {                   
                    updateTreeDepthAndIndex(child,node.getTreeDepth()+1);
                }
            
        }
        public int getTotalNodeCount()
        {
            return this.totalNodeCount;
        }
        public int getTotalLeafCount()
        {
            return this.totalLeafCount;
        }
       
        public Node getRoot()
        {
            return this.root;
        }
        public void assignInstancesToLeaf()
        {
            for(int i=0;i<dataset.numInstances();i++)
            {
                Node leaf=findLeafForInstance(dataset.get(i));
                leaf.addInstance(i);
            }
        }
        
        public Node findLeafForInstance(Instance inst)
        {
           leafFoundFlag=false;
           if(this.root!=null)
           {
               searchTree(inst,this.root);
           }
           return findLeaf;
        }
        public String getClassValueForInstance(Instance inst)
        {
           Node foundNode=findLeafForInstance(inst);
           return foundNode.getLeafPrediction();
        }
        private void searchTree(Instance inst,Node node)
        {
            if(node.isLeaf())
            {
                findLeaf=node;
                leafFoundFlag=true;
            }
            else
            {
                int ai=dataset.attribute(node.getSplitName()).index();
                double rval=inst.value(ai);
                List<Node> children=node.getChildren();
                for (int i=0;i<children.size() && leafFoundFlag==false;i++) 
                {
                    Node child =children.get(i);
                    String splitValue=child.getSplitValue();
                    //if(dataset.attribute(node.getSplitName()).isNumeric())
                    if(node.isNumeric())    
                    {
                        boolean flag=false;
                        double sval=Double.parseDouble(splitValue);
                        String splitOp=child.getSplitOp();
                        if(splitOp.equals("<="))
                           {
                               if(rval<=sval) flag=true;
                           }
                        else if(splitOp.equals("<"))
                           {
                               if(rval<sval) flag=true;
                           }                
                        else if(splitOp.equals(">"))
                           {
                               if(rval>sval) flag=true;
                           }
                        else if(splitOp.equals(">="))
                           {
                               if(rval>=sval) flag=true;
                           }
                        if(flag)
                        {
                            searchTree(inst,child);
                        }
                    }
                    else
                    {
                        String currentStr=inst.attribute(ai).value((int)rval);
                        if(splitValue.equals(currentStr))
                        {
                            searchTree(inst,child);
                        }
                    }
                }
                
            }
        }
        public void describeSubTree(Node node, StringBuilder out, int indent)
        {
            if(node.isLeaf())
            {
//                out.append(": "+node.getLeafPrediction()+" "+node.getClassDistribution());
                out.append(": "+node.getLeafPrediction()+" ("+node.getLeafSize()+")");
            }
            else
            {
                List<Node> children=node.getChildren();
                for (Node child : children) {
                    out.append("\n");
                    for(int j=0;j<indent;j++)
                    {
                        out.append(levelPadding);
                    }
                    out.append(node.getSplitName()+" "+child.getSplitOp()+" "+child.getSplitValue());
                    describeSubTree(child,out, child.getTreeDepth());
                }
            }
        }
        @Override
        public String toString()
        {
            StringBuilder out=new StringBuilder();
            describeSubTree(root, out,0);
            return out.toString();
        }
    }
    
    public class Node implements Serializable 
    {
        private static final long serialVersionUID = 5260380811391105058L;
        private Node parent;
        private List<String> testAttributes = new ArrayList<>();
        private List<Node> children = new ArrayList<>();
        private List<Integer> subdataset = new ArrayList<>();
        private boolean isLeafNode;
        private int nodeIndex;
        private int nodeType;
        private int treeDepth;
        private String splitAttrName;
        private String splitOp;
        private String splitValue;
        private String majorityClassValue;
        private String classValuesDistribution;
        private String []leafClassValues;
        private int []leafClassDistribution;
        private double confidence;
        
        Node(Node parent,int treeDepth,boolean isLeafNode,int nodeIndex)
        {
           this.parent=parent; 
           this.treeDepth=treeDepth;
           this.isLeafNode=isLeafNode;
           this.nodeIndex=nodeIndex;
           this.nodeType=2;
           leafClassValues=null;
        }

        public void setNodeInfo(int nodeType, String nodeName)
        {
            this.nodeType=nodeType;
            this.splitAttrName=nodeName;
        }
        public void setSplitInfo(String splitOp,String splitValue)
        {
            this.splitOp=splitOp;
            this.splitValue=splitValue;
        }
        public int getTreeDepth()
        {
            return this.treeDepth;
        }
        public void setTreeDepth(int treeDepth)
        {
           this.treeDepth=treeDepth;
        }
        public void setTestAttributes(List<String> testAttributes)
        {
            this.testAttributes.addAll(testAttributes);
        }
        public List<String>  getTestAttributes()
        {
            return this.testAttributes;
        }
        public String getSplitName()
        {
            return this.splitAttrName;
        }
        public String getSplitOp()
        {
            return this.splitOp;
        }
        public String getSplitValue()
        {
            return this.splitValue;
        }
        public boolean isNumeric()
        {
            return this.nodeType==1;
        }
        public void setLeafPrediction(String mCV)
        {
            this.majorityClassValue=mCV;
        }
        public String getLeafPrediction()
        {
            return this.majorityClassValue;
        }
        public void addChild(Node child)
        {
            this.children.add(child);
        }
        public void replaceChild(Node newChild, int index)
        {
            this.children.set(index, newChild);
        }
        public List<Node> getChildren()
        {
            return this.children;
        }
        public int getNumberOfChildren()
        {
             return this.children.size();
        }
        public void setParent(Node parent)
        {
           this.parent=parent;
        }
        public Node getParent()
        {
            return this.parent;
        }
        public Node getFirstChild()
        {
            if(this.children.size()>0)
                return this.children.get(0);
            else
                return null;
        }
        public Node getLastChild()
        {
            int noc=this.children.size();
            if(noc>0)
                return this.children.get(noc-1);
            else
                return null;
        }
        public boolean isLeaf()
        {
            return this.isLeafNode;
        }
        
        public void setNodeIndex(int nodeIndex)
        {
            this.nodeIndex=nodeIndex;            
        }
        public void addInstance(int instNo)
        {
            subdataset.add(instNo);
        }
        public List<Integer> getLeafData()
        {
           return subdataset;
        }
        public int getLeafData(int index)
        {
           return subdataset.get(index);
        } 
        public int getLeafSize()
        {
           return subdataset.size();
        } 
        public int getNodeIndex()
        {
            return this.nodeIndex;            
        }
        public void setClassDistribution(String []classValues, int []classDistribution)
        {
            this.leafClassValues=classValues.clone();
            this.leafClassDistribution=classDistribution.clone();
            classValuesDistribution="{";
            for(int i=0;i<leafClassValues.length;i++)
            {
                if(i==0)
                    classValuesDistribution+=leafClassValues[i]+":"+leafClassDistribution[i];
                else
                    classValuesDistribution+=", "+leafClassValues[i]+":"+leafClassDistribution[i];
            }
            classValuesDistribution+="}";
        }
        public boolean isClassDistributionSet()
        {
            if (this.leafClassValues==null)
            {
                return false;                
            }
            else{
                if (this.leafClassValues.length>0)
                    return true; 
                else
                    return false; 
            }
        }
        public String getClassDistribution()
        {            
            return this.classValuesDistribution;
        }
        public String []getLeafClassValues()
        {
            return this.leafClassValues.clone();
        }
        public int []getLeafClassDistribution()
        {
            return this.leafClassDistribution.clone();
        }
        public double getConfidence()
        {
            return this.confidence;
        }
        public void setConfidence(double confidence)
        {
            this.confidence=confidence;
        }
        public boolean isPure()
        {   
            return this.confidence==1.0;
        }
        
        @Override
        public String toString()
        {
            String nStr="";
            nStr+="Node Index:"+this.nodeIndex+", leaf size:"+getLeafSize()+"\n";
            nStr+="Class distribution:\n"+getClassDistribution();
            nStr+="\nIsPure:"+isPure()+", Class prediction:"+this.majorityClassValue;
            nStr+=", Confidence:"+this.confidence+"\n";
            return nStr;
        }
    }
}
