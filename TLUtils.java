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
 *   TLUtils.java  
 *   A Framework for Supervised Heterogeneous Transfer Learning 
 *   using Dynamic Distribution Adaptation and Manifold Regularization
 *
 *   @author Md Geaur Rahman and Md Zahidul Islam
 *   School of Computing, Mathematics & Engineering
 *   Charles Sturt University, Bathurst, NSW, Australia
 */
package transferlearning;


import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.SysFor;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import Jama.Matrix;
import Jama.QRDecomposition;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author Md Geaur Rahman <gea.bau.edu.bd>
 */
public class TLUtils {
    public static int findMinLeafSize(int numRecords)
    {
        if(numRecords<=20)
            return 2;
        else if(numRecords<=100)
            return (int)numRecords/10;
        else if(numRecords<=10000)
            return (int)Math.sqrt(numRecords);
        else if(numRecords<=20000)
            return 150;
        else if(numRecords<=40000)
            return 200;
        else if(numRecords<=50000)
            return 300;
        else if(numRecords<=75000)
            return 400;
        else if(numRecords<=100000)
            return 500;
        else if(numRecords<=150000)
            return 700;
        else if(numRecords<=200000)
            return 800;
        else if(numRecords<=400000)
            return 900;
        else if(numRecords<=500000)
            return 1000;
        else 
            return 2000;
    }
    public static double calculateAccuracy(String modelFile,String testFile)
    {
        double acc=0;
         try{
             
            Classifier model = loadModel(modelFile);
            if(model!=null)
                acc=calculateAccuracy(model,testFile);
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        } 
         return acc;
    }
    
    public static double calculateAccuracy(Classifier model, String testFile)
    {
        double acc=0;
         try{
            ConverterUtils.DataSource source=new ConverterUtils.DataSource(testFile);
            Instances testdata=source.getDataSet();
            acc=calculateAccuracy(model,testdata);
         }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        } 
         return acc;
    }
    
    public static double calculateAccuracy(Classifier model,Instances testdata)
    {
        double acc=0;
         try{
            testdata.setClassIndex(testdata.numAttributes()-1);
            int total=testdata.numInstances();
            int correct=0;
            for (int j=0;j<testdata.numInstances();j++){
                    double actualClass = testdata.instance(j).classValue();
                    String actual = testdata.classAttribute().value((int) actualClass);
                    Instance newInst = testdata.instance(j);
                    double preNB = model.classifyInstance(newInst);
                    String predString = testdata.classAttribute().value((int) preNB);
                    if(actual.equals(predString))correct++;
            }
            acc=100.0*correct/total;
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        } 
         return acc;
    }
    
    public static Classifier loadModel(String modelFile)
    {   
        Classifier model=null;
        try{
             ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile));
            model = (Classifier) ois.readObject();
            ois.close();            
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        } 
        return model;
    }
    public static void saveModel(Classifier model,String modelFile)
    {
        try{
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelFile));
            oos.writeObject(model);
            oos.flush();
            oos.close();
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }  
    }
    public static void writeARRF(Instances data, String file)
    {
        try{
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(file));
            saver.writeBatch();
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }
    }
    public static Classifier buildSingleForest(String trainFile, int minRecords)
    {
        Classifier cls=null;
         try{
            ConverterUtils.DataSource source=new ConverterUtils.DataSource(trainFile);
            Instances dataset=source.getDataSet();
            cls=buildClassifier(dataset,1,minRecords,0);            
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }    
        return cls;
    }
    public static Classifier buildSingleForest(String trainFile, int minRecords, int method)
    {
        Classifier cls=null;
         try{
            ConverterUtils.DataSource source=new ConverterUtils.DataSource(trainFile);
            Instances dataset=source.getDataSet();
            cls=buildClassifier(dataset,1,minRecords,method);            
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }    
        return cls;
    }
    
    public static void buildSingleForest(String trainFile, int minRecords, String modelFile, int method)
    {
        try{
            ConverterUtils.DataSource source=new ConverterUtils.DataSource(trainFile);
            Instances dataset=source.getDataSet();
            Classifier cls=buildClassifier(dataset,1,minRecords,method);  
            saveModel(cls,modelFile);            
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }    
    }
    public static void buildSingleForest(String trainFile, int minRecords, String modelFile)
    {
        buildSingleForest(trainFile,minRecords,modelFile,0);
    }
    public static void buildClassifier(String trainFile, int numTrees, int minRecords, String modelFile)
    {
        buildClassifier(trainFile,numTrees,minRecords,modelFile,0);         
    }
    public static void buildClassifier(String trainFile, int numTrees, int minRecords,
            String modelFile, int method)
    {
        Classifier cls=null;
         try{
            ConverterUtils.DataSource source=new ConverterUtils.DataSource(trainFile);
            Instances dataset=source.getDataSet();
            cls=buildClassifier(dataset,numTrees,minRecords,method);  
            saveModel(cls,modelFile);   
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }    
    }
    public static Classifier buildClassifier(String trainFile, int numTrees, int minRecords)
    {
        return buildClassifier(trainFile,numTrees,minRecords,0);
    }
    public static Classifier buildClassifier(String trainFile, int numTrees, int minRecords, int method)
    {
        Classifier cls=null;
         try{
            ConverterUtils.DataSource source=new ConverterUtils.DataSource(trainFile);
            Instances dataset=source.getDataSet();
            cls=buildClassifier(dataset,numTrees,minRecords,method);            
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }    
        return cls;
    }
    
    
    public static Classifier buildClassifier(Instances dataset, int numTrees, int minRecords, int method)
    {
        
        Classifier cls=null;
         try{
            dataset.setClassIndex(dataset.numAttributes()-1);
            if(method==1)//SysFor
            {
                SysFor sysFor=new SysFor();
                sysFor.setMinRecLeaf(minRecords);
                sysFor.setNumberTrees(numTrees);
                sysFor.setGoodness(0.3f);
                sysFor.setConfidence(0.25f);
                sysFor.setSeparation(0.3f);            
                sysFor.buildClassifier(dataset);
    //            String tree=sysFor.toString();
                cls=(Classifier) sysFor;
    //            System.out.println(tree);
            }
            else  //RandomForest
            {
                String []options={"-P","100","-print","-I","1","-num-slots","1",
                    "-K","0","-M","2","-V","0.001","-S","1"};
                options[4] = numTrees+"";
                options[10] = minRecords+"";
                RandomForest rf=new RandomForest();
                rf.setOptions(options);
                rf.buildClassifier(dataset);
                cls=(Classifier) rf;
            }
        }
        catch(Exception e)
        {
            System.out.println("Error: "+e);
        }    
        return cls;
    }
    public static String majorityClassValue(Instances allData)
    {
        String []classNames = new String[allData.numClasses()];
        for (int i = 0; i < allData.numClasses(); i++) {
            classNames[i] = allData.classAttribute().value(i);            
        }
        int[] classCounts=new int[allData.numClasses()];
        for(Instance ins: allData)
             classCounts[(int)(ins.classValue())]++;
        int index=0;
        for (int i = 1; i < allData.numClasses(); i++) {
           if(classCounts[i]>classCounts[index]){
               index=i;
           }
        }
        return classNames[index];
    }
        /** Take a certain percentage of a set of instances.
	 * @param instances
	 * @param percentage
	 * @return a reduced set of instances according to the given percentage
	 */
	public static Instances trimInstances(Instances instances, double percentage) {
		int numInstancesToKeep = (int) Math.ceil(instances.numInstances() * percentage);
		return trimInstances(instances, numInstancesToKeep);
	}

	/** Take a certain number of a set of instances.
	 * @param instances
	 * @param numInstances the number of instances to keep
	 * @return a reduced set of instances according to the given number to keep
	 */
	public static Instances trimInstances(Instances instances, int numInstances) {
		Instances trimmedInstances = new Instances(instances);
		for (int i = trimmedInstances.numInstances() - 1; i >= numInstances; i--) {
			trimmedInstances.delete(i);
		}
		return trimmedInstances;
	}
	
	/** Extract a particular subset of the instances.
	 * @param instances
	 * @param startIdx the start instance index
	 * @param numInstancesToRetrieve the number of instances to retrieve
	 * @return the specified subset of the instances.
	 */
	public static Instances subsetInstances(Instances instances, int startIdx, int numInstancesToRetrieve) {
		double possibleNumInstancesToRetrieve = instances.numInstances()-startIdx;
		if (numInstancesToRetrieve > possibleNumInstancesToRetrieve) {
			throw new IllegalArgumentException("Cannot retrieve more than "+possibleNumInstancesToRetrieve+" instances.");
		}
		
		int endIdx = startIdx + numInstancesToRetrieve - 1;
		
		// delete all instance indices outside of [startIdx, endIdx]
		Instances subset = new Instances(instances);
		for (int i=subset.numInstances()-1; i>= 0; i--) {
			if (i < startIdx || i > endIdx)
				subset.delete(i);
		}
		
		return subset;
	}


	/** Merge two instance sets.
	 * @param instances1
	 * @param instances2
	 * @return the merged instance sets
	 */
	public static Instances mergeInstances(Instances instances1, Instances instances2) {
		if(instances1 == null)
			return instances2;
		if(instances2 == null)
			return instances1;
		if (!instances1.checkInstance(instances2.firstInstance()))
			throw new IllegalArgumentException("The instance sets are incompatible.");
		Instances mergedInstances = new Instances(instances1);
		Instances tempInstances = new Instances(instances2);
		for (int i = 0; i < tempInstances.numInstances(); i++) {
			mergedInstances.add(tempInstances.instance(i));
		}
		return mergedInstances;
	}
        /** check domain values of projected data and test data.
	 * @param instances1
	 * @param instances2
	 * @return the compitable test data
	 */
	public static Instances checkProjectedAndTestData(Instances proj, Instances test) {
		if(proj == null)
			return test;

		String relation=test.relationName();
                int numAttr=test.numAttributes();
                ArrayList<Attribute> testAtts= new ArrayList<Attribute>();
                ArrayList<String> []testVals=new ArrayList[numAttr];
                for(int i=0;i<numAttr;i++)
                {
                    if(test.attribute(i).isNominal())
                    {
                        testVals[i] = new ArrayList<String>();
                        int numv=test.attribute(i).numValues();
                        for(int k=0;k<numv;k++)
                            testVals[i].add(test.attribute(i).value(k));
                        numv=proj.attribute(i).numValues();
                        for(int k=0;k<numv;k++)
                        {
                            String dv=proj.attribute(i).value(k);
                            if(!testVals[i].contains(dv))
                                testVals[i].add(dv);
                        }
                        testAtts.add(new Attribute(test.attribute(i).name(),testVals[i]));
                    }
                    else{
                        testAtts.add(new Attribute(test.attribute(i).name()));
                    }                        
                }
                 
                int t=test.numInstances();               
                double [][]alldata=new double[t][numAttr];   
                //src data
                for(int i=0;i<t;i++)
                {
                    Instance inst=test.get(i);
                    for(int j=0;j<numAttr;j++)
                    {
                        if(test.attribute(j).isNominal())
                        {
                            double v=inst.value(j);
                            String tv=test.attribute(j).value((int)v);
                            alldata[i][j]=testVals[j].indexOf(tv);
                        }
                        else
                        {
                            alldata[i][j]=inst.value(j);
                        }                        
                    }
                }
               
                Instances checkTestInstances = new Instances(relation, testAtts, 0);
                for(int i=0;i<t;i++)
                {
                    checkTestInstances.add(new DenseInstance(1.0, alldata[i]));
                }	
                checkTestInstances.setClassIndex(test.classIndex());
		return checkTestInstances;
	}
        /** Merge two instance sets of two similar (not same) domains, 
         *  but they have the same attributes.
	 * @param src
	 * @param tgt
	 * @return the merged instance sets
	 */
	public static Instances mergeInstancesTL(Instances src, Instances tgt) {
		if(src == null)
			return tgt;
		if(tgt == null)
			return src;
		String relation=src.relationName();
                int numAttr=src.numAttributes();
                ArrayList<Attribute> testAtts= new ArrayList<Attribute>();
                ArrayList<String> []testVals=new ArrayList[numAttr];
                for(int i=0;i<numAttr;i++)
                {
                    if(src.attribute(i).isNominal())
                    {
                        testVals[i] = new ArrayList<String>();
                        int numv=src.attribute(i).numValues();
                        for(int k=0;k<numv;k++)
                            testVals[i].add(src.attribute(i).value(k));
                        numv=tgt.attribute(i).numValues();
                        for(int k=0;k<numv;k++)
                        {
                            String dv=tgt.attribute(i).value(k);
                            if(!testVals[i].contains(dv))
                                testVals[i].add(dv);
                        }
                        testAtts.add(new Attribute(src.attribute(i).name(),testVals[i]));
                    }
                    else{
                        testAtts.add(new Attribute(src.attribute(i).name()));
                    }                        
                }
                
                int s=src.numInstances();
                int t=tgt.numInstances();
                int total=s+t;
                double [][]alldata=new double[total][numAttr];   
                //src data
                for(int i=0;i<s;i++)
                {
                    Instance inst=src.get(i);
                    for(int j=0;j<numAttr;j++)
                    {
                        if(src.attribute(j).isNominal())
                        {
                            double v=inst.value(j);
                            String tv=src.attribute(j).value((int)v);
                            alldata[i][j]=testVals[j].indexOf(tv);
                        }
                        else
                        {
                            alldata[i][j]=inst.value(j);
                        }                        
                    }
                }
                //merging data
                for(int i=0,r=s;i<t && r<total;i++,r++)
                {
                    Instance inst=tgt.get(i);
                    for(int j=0;j<numAttr;j++)
                    {
                        if(src.attribute(j).isNominal())
                        {
                            double v=inst.value(j);
                            String tv=tgt.attribute(j).value((int)v);
                            alldata[r][j]=testVals[j].indexOf(tv);
                        }
                        else
                        {
                            alldata[r][j]=inst.value(j);
                        }                        
                    }
                }
                Instances mergedInstances = new Instances(relation, testAtts, 0);
                for(int i=0;i<total;i++)
                {
                    mergedInstances.add(new DenseInstance(1.0, alldata[i]));
                }		
		return mergedInstances;
	}
        
        /** change the class values with cv.
	 * @param instances
	 * @param cv
	 * @return the changed instance sets
	 */
	public static Instances changeClassValue(Instances instances, String cv) {
		
                String relation=instances.relationName();
                int numAttr=instances.numAttributes();
                ArrayList<Attribute> testAtts= new ArrayList<Attribute>();
                ArrayList<String> []testVals=new ArrayList[numAttr];
                for(int i=0;i<numAttr;i++)
                {
                    if(instances.attribute(i).isNominal())
                    {
                        testVals[i] = new ArrayList<String>();
                        if(i==instances.classIndex())
                        {
                             testVals[i].add(cv);
                        }
                        else{
                            int numv=instances.attribute(i).numValues();
                            for(int k=0;k<numv;k++)
                                testVals[i].add(instances.attribute(i).value(k));
                        }
                        testAtts.add(new Attribute(instances.attribute(i).name(),testVals[i]));
                        
                    }
                    else{
                        testAtts.add(new Attribute(instances.attribute(i).name()));
                    }                        
                }
                
                int s=instances.numInstances();
                double [][]alldata=new double[s][numAttr];   
                //src data
                for(int i=0;i<s;i++)
                {
                    Instance inst=instances.get(i);
                    for(int j=0;j<numAttr;j++)
                    {
                        if(instances.attribute(j).isNominal())
                        {
                            if(j==instances.classIndex())
                            {
                                alldata[i][j]=testVals[j].indexOf(cv);
                            }
                            else{
                                double v=inst.value(j);
                                String tv=instances.attribute(j).value((int)v);
                                alldata[i][j]=testVals[j].indexOf(tv);
                            }
                        }
                        else
                        {
                            alldata[i][j]=inst.value(j);
                        }                        
                    }
                }
                
                Instances newInstances = new Instances(relation, testAtts, 0);
                for(int i=0;i<s;i++)
                {
                    newInstances.add(new DenseInstance(1.0, alldata[i]));
                }		
		return newInstances;
	}
        
         /** remove attributes.
	 * @param instances
	 * @param indices
	 * @return the final instance sets
	 */
	public static Instances removeAttributes(Instances instances, String indices) throws Exception{
            Remove remove = new Remove();
            remove.setAttributeIndices(indices);
            remove.setInputFormat(instances);
            Instances instNew = Filter.useFilter(instances, remove);
            return instNew;
	}
         /** remove an attribute.
	 * @param instances
	 * @param comAttr
         * @param index
	 * @return the final instance sets
	 */
	public static Instances removeAttributes(Instances instances, int comAttr[][],int index){
            String indices="";
            boolean first=true;
            for(int k=0;k<instances.numAttributes();k++)
            {
                boolean found=false;
                for(int i=0;i<comAttr.length;i++)
                {
                    if(k==comAttr[i][index])
                    {
                        found=true;break;
                    }
                }
                if(!found)
                {
                    if(first)
                    {
                        indices=(k+1)+"";
                        first=false;
                    }
                    else{
                        indices=","+(k+1);
                    }
                }
            }
            
            Instances instNew=null;
            try{
                instNew =removeAttributes(instances, indices);
            }
            catch(Exception e)
            {

            }
            return instNew;
	}
        
        
	/**
	 * Converts an instance to a feature vector excluding the class attribute.
	 * @param instance The instance.
	 * @return A vector representation of the instance excluding the class attribute
	 */
	public static double[] instanceToDoubleArray(Instance instance) {
		double[] vector = new double[(instance.classIndex() != -1) ? instance.numAttributes() - 1 : instance.numAttributes()];
		double[] instanceDoubleArray = instance.toDoubleArray();
		int attIdx = 0;
		for (int i = 0; i < vector.length; i++) {
			if (i == instance.classIndex()) {
				attIdx++;
			}
			vector[i] = instanceDoubleArray[attIdx++];
		}
		return vector;
	}


	/**
	 * Converts a set of instances to an array of vectors
	 * @param instances The set of instances.
	 * @return The array of feature vectors.
	 */
	public static double[][] instancesToDoubleArrays(Instances instances) {
		double[][] vectors = new double[instances.numInstances()][];
		for (int instIdx = 0; instIdx < instances.numInstances(); instIdx++) {
			vectors[instIdx] = instanceToDoubleArray(instances.instance(instIdx));
		}
		return vectors;
	}
        
        public static Instances doubleArraysToInstances(Instances instances, double [][]values)
        {
            Instances data=new Instances(instances,0);
            for(int i=0;i<values.length;i++)
            {
                data.add(new DenseInstance(1.0, values[i]));
            }
            return data;
        }
	
   public static String preprocessTree(String treeStr)
        {            
            String []rules=treeStr.split("\n");
            String rec="";int t=0;
            for(int i=0;i<rules.length;i++)
            {
                if(rules[i].equals("RandomTree"))
                {
                    t++;
                    rec=rec+"Tree "+t+":\n";i++;
                   while(i<rules.length)
                   {
                       if(rules[i].equals("RandomTree"))
                       {
                                   i--;break;
                       }
                       else
                       {
                           int f=0;
                           if(rules[i].length()>16)
                           {
                               if(rules[i].substring(0, 16).equals("Size of the tree"))f=1;
                           }
                           if(f==0&&!rules[i].equals("") && !rules[i].equals("=========="))
                           {
                               rec=rec+rules[i]+"\n";
                           }
                           i++;
                       }
                   }
                }
            }
            return rec;
        }
    
   public static String processConditionPart(String con)
   {
       String var=con.substring(con.indexOf(":")+1, con.indexOf("]"));
       int l=con.indexOf("]")+1;
       String op=con.substring(l, l+3).trim();
       String val=con.substring(con.indexOf(op)+op.length(), con.length()-1).trim();
       if(val.contains(":")){
           val=val.substring(val.indexOf(":")+1, val.indexOf("}"));
       }
       return var+" "+op+" "+val;
   }
   public static String processLeafPart(String con)
   {
       String rec=con.substring(con.indexOf("<")+1, con.indexOf(">"));
       return rec.substring(rec.indexOf(":")+1);
   }
   
   //Find the number of conditions of each trees
     public static String[] processTrees(String []trees, int []Loc)
        {
            int numTree=Loc.length;
            for(int i=0;i<numTree;i++)
            {
                Loc[i]=0;
            }
            int count=0;
            int pos=0;
            int tlines=trees.length;
            ArrayList<String> Conditions=new ArrayList<String>();
            for(int i=0;i<tlines;i++)
            {
                if(!trees[i].equals("")){
                if(trees[i].startsWith("Tree"))
                {
               
                    Loc[count]=pos;
                    count++;    
                }
                else
                {
                    String []cac=separateConditionClass(trees[i]);
                    if(!cac[1].isEmpty())
                     {
                         StringTokenizer tokenizer= new StringTokenizer(cac[1], " {}():\t\n\r\f");
                         if(tokenizer.hasMoreTokens()) cac[1]=tokenizer.nextToken();
                         cac[0]=cac[0].trim()+": "+cac[1];
                     }                
                    Conditions.add(cac[0]);
                    pos++;
                }
                }
            }
            return Conditions.toArray(new String[0]);
        }
     
     //count the number of trees
  public static int countTrees(String []trees)
    {
        int count=0;
        int tlines=trees.length;
        for(int i=0;i<tlines;i++)
        {
            if(!trees[i].equals("") && trees[i].length()>4)
            {
                if(trees[i].substring(0, 4).equals("Tree"))
                {
                    count++;
                }
            }
        }
        return count;
    }
    public static boolean isConditionALeaf(String condition)
        {
                return condition.contains(":");
        }
        public static String [] separateConditionClass(String condition)
        {
            String []cac=new String[2];
            cac[0]=condition;
            cac[1]="";
            if(isConditionALeaf(condition))
            {
                 cac[1]=condition.substring(condition.indexOf(":")+1, condition.length());
                 cac[1]=cac[1].trim();
                 cac[0]=cac[0].substring(0, cac[0].indexOf(":"));
                 cac[0]=cac[0].trim();
            }
            return cac;
        }
        public static String[]convertStringToArray(String strVal) 
        {
            ArrayList<String> arrVal=new ArrayList<String>();
            StringTokenizer tokenizer= new StringTokenizer(strVal, " :\t\n\r\f");
            int n=tokenizer.countTokens();
            for(int i=0;i<n;i++)
            {
                arrVal.add(tokenizer.nextToken());
            }
            return arrVal.toArray(new String[0]);
        }    
        public static int findAttrType(String currentAttr, Instances dataset)
        {
            if(dataset.attribute(currentAttr).isNumeric())
                return 1;
            else
                return 0;
        }
    //find shared contribution
    public static int[][]findPivots(double [][]JSD, double similarityThreshold)
    {
        ArrayList<Integer> sourcePivot=new ArrayList<Integer>(); 
        ArrayList<Integer> targetPivot=new ArrayList<Integer>(); 
        for(int i=0;i<JSD.length;i++)
        {
            int p=-1;
            for(int j=0;j<JSD[i].length;j++)
            {
                if(JSD[i][j]>=similarityThreshold)
                {
                    if(p==-1)
                        p=j;
                    else if(JSD[i][j]>JSD[i][p])
                        p=j;
                }
            }
            if(p>=0){
                if(!sourcePivot.contains(i)&& !targetPivot.contains(p)){
                sourcePivot.add(i);
                targetPivot.add(p);
                }
            }
        }
        int np=sourcePivot.size();
        int [][]pivots=new int[np][2];
        for(int i=0;i<np;i++)
        {
            pivots[i][0]=sourcePivot.get(i);
            pivots[i][1]=targetPivot.get(i);
        }
        return pivots.clone();
    }
    
    //find pivot attribute contribution
    public static double [][]findAttributeContribution(double [][]contribution,int [][]pivots, int domain)
    {
        int np= pivots.length;
        int noa=contribution[0].length;
        double [][]st=new double[np][noa];
        for(int i=0;i<np;i++)
        {
            int r=pivots[i][domain];
            for(int c=0;c<noa;c++)
            {
                st[i][c]=contribution[r][c];
            }
        }
        return st.clone();
    }
    
    
    //find pivot attribute contribution
    public static String [][]findAttributeContribution(String [][]contribution,int [][]pivots, int domain)
    {
        int np= pivots.length;
        int noa=contribution[0].length-1;
        String [][]st=new String[np][noa];
        for(int i=0;i<np;i++)
        {
            int r=pivots[i][domain];
            for(int c=0;c<noa;c++)
            {
                st[i][c]=contribution[r][c];
            }
        }
        return st.clone();
    }
    
   
    
    public static void displayProjection(double[][]projection)
    {
            System.out.println("\nProjection:");
            for (int j = 0; j < projection.length; j++) {                
                for(int t=0; t<projection[j].length;t++)
                {
                   if(t==projection[j].length-1)
                      System.out.println(projection[j][t]); 
                   else
                       System.out.print(projection[j][t]+", ");
                }
            }
    }
    public static void display(int[][]projection, String msg)
    {
            System.out.println("\n"+msg+":");
            for (int j = 0; j < projection.length; j++) {                
                for(int t=0; t<projection[j].length;t++)
                {
                   if(t==projection[j].length-1)
                      System.out.println(projection[j][t]); 
                   else
                       System.out.print(projection[j][t]+", ");
                }
            }
    }
    public static void display(String[][]projection, String msg)
    {
            System.out.println("\n"+msg+":");
            for (int j = 0; j < projection.length; j++) {                
                for(int t=0; t<projection[j].length;t++)
                {
                   if(t==projection[j].length-1)
                      System.out.println(projection[j][t]); 
                   else
                       System.out.print(projection[j][t]+", ");
                }
            }
    }
    public static void display(double[][]projection, String msg)
    {
            System.out.println("\n"+msg+":");
            for (int j = 0; j < projection.length; j++) {                
                for(int t=0; t<projection[j].length;t++)
                {
                   if(t==projection[j].length-1)
                      System.out.println(projection[j][t]); 
                   else
                       System.out.print(projection[j][t]+", ");
                }
            }
    }
    
     public static void display(double[][]centroids, int k)
        {
            System.out.println("\nCentrods:");
            for (int j = 0; j < k; j++) {
                double []x=centroids[j];
                for(int t=0; t<x.length;t++)
                {
                   if(t==x.length-1)
                      System.out.println(x[t]); 
                   else
                       System.out.print(x[t]+", ");
                }
            }
        }
  
     
   
   
    public static double[][] normalize(double[][] arr)
    {        
        int row=arr.length;
        int col=arr[0].length;
        for (int j = 0; j < col; j++) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_VALUE;
            for (int i = 0; i < row; i++) {
                max = Math.max(arr[i][j],max);
                min = Math.min(arr[i][j],min);
            }
            for (int i = 0; i < row; i++) {
                arr[i][j] = (arr[i][j] - min)/(max-min);
            }
        }
        return arr;
    }
    
    public static double [][]ManifoldMMDMLR(double [][]source, double [][]target, double sigma, double manireg)
    {
        int scol=source[0].length;
        int trow=target.length;
        int tcol=target[0].length;
        double [][]projection=new double[scol][tcol];
        for(int tc=0;tc<tcol;tc++)
        {
            double []y=new double[trow];
            boolean allValueZero=true;
            for(int tr=0;tr<trow;tr++)
            {
                y[tr]=target[tr][tc];
                if(y[tr]!=0)allValueZero=false;
            }
            double []w;
            if(allValueZero){
                w=new double[scol];
                Arrays.fill(w,0);
            }
            else{
                w=MMR(source,y);
                
                
//                try{
//                ManifoldReg manifoldr=new ManifoldReg(source.clone(),y.clone(),sigma,manireg);            
//                w=manifoldr.coefficients();
//                }
//                catch(Exception e){
//                    w=new double[scol];
//                    Arrays.fill(w,0);
//                }
            }
            if(w.length==scol)
            {
                for(int sr=0;sr<scol;sr++)
                {
                    if(w[sr]>=0)projection[sr][tc]=w[sr];
                }
            }
        }
        return projection.clone();
    }
    
    
    
    public static double[]MMR(double[][] x, double[] y)
    {
        if (x.length != y.length) {
            throw new IllegalArgumentException("matrix dimensions don't agree");
        }

        // number of observations
        int n = y.length;

        Matrix matrixX = new Matrix(x);

        // create matrix from vector
        Matrix matrixY = new Matrix(y, n);

        // find least squares solution
        QRDecomposition qr = new QRDecomposition(matrixX);
        Matrix beta = qr.solve(matrixY);
        int m=x[0].length;
        double []coefficients=new double [m];
        for(int i=0;i<m;i++)
            coefficients[i]=beta.get(i, 0);
        return coefficients;
    }
    
    
    public static double calculateManifoldRegularizer(double [][]source, double [][]target, double lampda, int [][]comAttr)
    {
        double reg=0;
        int nps=source.length;
        int npt=target.length;
        int np=nps+npt;
        double p1=lampda/(np*np);
        double p2=1;
        int cl=comAttr.length;
        double [][]source_transpose=arrayTranspose(source.clone());
        if(cl>0)
        {
            double [][]W=calculateDistance(source,target,comAttr);
            double [][]L=calculateL(W);
            double [][]source_transposeL=arrayMultiplication(source_transpose,L);
            double [][]source_transposeLTarget=arrayMultiplication(source_transposeL,target);
            p2=calculateEuclideanNorm(source_transposeLTarget);
        }
        else
        {
            double [][]source_transposeTarget=arrayMultiplication(source_transpose,target);
            p2=calculateEuclideanNorm(source_transposeTarget);
        }
        if(p2>0)
            reg=p1*p2;
        else
            reg=p1;
        return reg;
    }
    
    public static double calculateEuclideanNorm(double [][]W)
    {
        int n=W.length;
        int m=W[0].length;
        double sum=0;
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {   
                sum+=W[i][j]*W[i][j];
            }
        }
        return Math.sqrt(sum);
    }
    public static double[][] calculateL(double [][]W)
    {
        int n=W.length;
        int m=W[0].length;
        //find L=D-W, where D is a diagonal matrix having d_ii=W_ij for all j
        double [][]L=new double[n][m];
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<m;j++)
            {   
                if(i==j)
                    L[i][j]=0;
                else
                    L[i][j]=0-W[i][j];
            }
        }
        return L.clone();
    }
    public static double[][] calculateDistance(double [][]source, double [][]target, int [][]comAttr)
    {
        int sr=source.length;
        int tr=target.length;
        int cl=comAttr.length;
        double [][]d=new double[sr][tr];
        if(cl>0)
        {
            for(int i=0;i<sr;i++)
            {
                for(int j=0;j<tr;j++)
                {   
                    d[i][j]=eucledianDistance(source[i],target[j],comAttr);
                }
            }
        }
        return d.clone();
    }
    public static double eucledianDistance(double[] point1, double[] point2, int [][]comAttr) 
    {
        double sum = 0.0;
        for(int i = 0; i < comAttr.length; i++) {
            sum += ((point1[comAttr[i][0]] - point2[comAttr[i][1]]) * (point1[comAttr[i][0]] - point2[comAttr[i][1]]));
        }
        return Math.sqrt(sum);
    }
    public static double[][] arrayMultiplication(double [][]a, double [][]b)
    {
        int arow=a.length;
        int acol=a[0].length;
        int brow=b.length;
        int bcol=b[0].length;
        double [][]c=new double[arow][bcol];
        if(acol==brow) //if multiplication rule is satisfied
        {
            for(int i=0;i<arow;i++)
            {
                for(int j=0;j<bcol;j++)
                {   
                    c[i][j]=0;
                    for(int k=0;k<acol;k++)
                    {
                        c[i][j]+=a[i][k]*b[k][j];
                    }
                }
            }
        }
        return c.clone();
    }
    public static double[][] arrayTranspose(double [][]s)
    {
        int row=s.length;
        int col=s[0].length;
        double [][]d=new double[col][row];
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                d[j][i]=s[i][j];
            }
        }
        return d.clone();
    }
    
    public static ArrayList<String> getClassValues(Instances dataset)
    {
        ArrayList<String> cv=new ArrayList<String>();
        int num=dataset.numClasses();
        for(int i=0;i<num;i++)
            cv.add(dataset.classAttribute().value(i));
        return cv;
    }
    public static double[][] calculateSimilarityMatrix(double [][]srcDist,double [][]tgtDist,
                    Instances src,Instances tgt)
    {
        ArrayList<String> scv=getClassValues(src);
        ArrayList<String> tcv=getClassValues(tgt);
        ArrayList<String> cv=new ArrayList<String>(); 
        //find all cv's
        int ncv=scv.size();
        for(int i=0;i<ncv;i++){
            cv.add(scv.get(i));
        }
        ncv=tcv.size();
        for(int i=0;i<ncv;i++){
            if(!cv.contains(tcv.get(i))){
                cv.add(tcv.get(i));
            }
        }
        //store src distribution
        int sleaf=srcDist.length;
        double [][]srcDistNew=new double[sleaf][cv.size()];
        for(int r=0;r<sleaf;r++) {
            for(int c=0;c<cv.size();c++){
                if(scv.contains(cv.get(c))){
                    srcDistNew[r][c]=srcDist[r][scv.indexOf(cv.get(c))];
                }
                else{
                    srcDistNew[r][c]=0;
                }
            }
        }
        //store target distribution
        int tleaf=tgtDist.length;
        double [][]tgtDistNew=new double[tleaf][cv.size()];
        for(int r=0;r<tleaf;r++){
            for(int c=0;c<cv.size();c++){
                if(tcv.contains(cv.get(c))){
                    tgtDistNew[r][c]=tgtDist[r][tcv.indexOf(cv.get(c))];
                }
                else{
                    tgtDistNew[r][c]=0;
                }
            }
        }
        
        
        //calculate similarity(=1-JSD)  
        double [][]similarity=new double[srcDistNew.length][tgtDistNew.length];
        for(int r=0;r<srcDistNew.length;r++){
            for(int c=0;c<tgtDistNew.length;c++){
                similarity[r][c]=1-calculateJSD(srcDistNew[r].clone(),tgtDistNew[c].clone());
            }
        }
        return similarity.clone();
    }
    public static double[][] calculateDistanceMatrix(double [][]srcDist,double [][]tgtDist)
    {
        //calculate JSD
        double [][]distance=new double[srcDist.length][tgtDist.length];
        for(int r=0;r<srcDist.length;r++)
        {
            for(int c=0;c<tgtDist.length;c++)
            {
                distance[r][c]=calculateJSD(srcDist[r].clone(),tgtDist[c].clone());
            }
        }
        return distance.clone();
    }
    public static double calculateJSD(double []p, double[]q)
    {
        double distance=0;
        double []m=new double[p.length];
        for(int i=0;i<p.length;i++)
        {
           m[i]=(p[i]+q[i])/2.0;     
        }
        distance=(klDivergence(p.clone(),m.clone())+klDivergence(q.clone(),m.clone()))/2.0;
        return distance;
    }
    public static double klDivergence(double []p, double[]q)
    {
        double distance=0;
        if(p.length==q.length)
        {
            for(int i=0;i<p.length;i++)
            {
                if(p[i]!=0&& q[i]!=0)
                    distance+=p[i]*(Math.log(p[i]/q[i])/Math.log(2.0));
            }
        }
        return distance;
    }
    
    public static double cosineSimilarity(double[] vectorA, double[] vectorB,int []attrType) 
    {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;
        for (int i = 0; i < vectorA.length; i++) {
            double a=vectorA[i];
            double b=vectorB[i];
            if(attrType[i]==0){
               if( a==b){
                   a=1;b=1;
               }
               else{
                   a=0;b=0;
               }
            }
            dotProduct += a * b;
            normA += Math.pow(a, 2);
            normB += Math.pow(b, 2);
        }
        if(normA<=0||normB<=0)
            return 0;
        else
            return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
    public static double[][] calculateCosineSimilarity(Instances data)
    {
        int n=data.numInstances();
        int noa=data.numAttributes();
        int []attrType=new int[noa];
        for(int i=0;i<noa;i++){
            if(data.attribute(i).isNumeric())
                attrType[i]=1;
            else
                attrType[i]=0;
        }
        double [][]val=instancesToDoubleArrays(data);
        double [][]cs=new double[n][n];
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++){
                if(i==j)
                    cs[i][j]=1;
                else{
                    double s=cosineSimilarity(val[i],val[j],attrType);
                    cs[i][j]=s;cs[j][i]=s;
                }
            }
        }
        return cs;
    }
    
    public static double kernelSimilarity(double[] vectorA, double[] vectorB,int []attrType) 
    {
        double distance = 0.0;
        double similarity = 0.0;
        int noa=vectorA.length;
        for (int i = 0; i < vectorA.length; i++) {
            if(attrType[i]==0){
               if( vectorA[i]!=vectorB[i]){
                       distance+=1;
               }}
            else{
                  distance+=Math.pow(vectorA[i]-vectorB[i],2);
               }
            

        }
        if(noa>0)similarity=1-(distance/noa);
        return similarity;
    }
    public static double[][] kernel(Instances data)
    {
        int n=data.numInstances();
        int noa=data.numAttributes();
        int []attrType=new int[noa];
        for(int i=0;i<noa;i++){
            if(data.attribute(i).isNumeric())
                attrType[i]=1;
            else
                attrType[i]=0;
        }
        Instances normData=null;
        try{
        Normalize m_Filter = new Normalize();
        normData = new Instances(data);
        m_Filter.setInputFormat(normData);        
        normData = Filter.useFilter(normData, m_Filter);
        }
        catch(Exception e){
            normData = new Instances(data);
        }
        double [][]val=instancesToDoubleArrays(normData);
        double [][]ks=new double[n][n];
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++){
                if(i==j)
                    ks[i][j]=1;
                else{
                    double s=kernelSimilarity(val[i],val[j],attrType);
                    ks[i][j]=s;ks[j][i]=s;
                }
            }
        }
        return ks;
    }
    public static double[][]rbfKernel()
    {
        int n=10;
        double [][]ks=new double[n][n];
        
        return ks;
    }
    public static double[][] identityMatrix(int n)
    {
        double [][]I=new double[n][n];
        for(int i=0;i<n;i++)
        {
           Arrays.fill(I[i], 0);
           I[i][i]=1;
        }
        return I;
    }
}
