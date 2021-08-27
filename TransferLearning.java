/*
 * TLF:  
 * A Framework for Supervised Heterogeneous Transfer Learning 
 * using Dynamic Distribution Adaptation and Manifold Regularization
 * 
 */
package transferlearning;


import com.mathworks.engine.MatlabEngine;
import java.io.File;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import transferlearning.RulesToForest.Node;

import transferlearning.utils.TLUtils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.Matrix;
import weka.core.neighboursearch.LinearNNSearch;

import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;

/**
 *
 * @author Md Geaur Rahman <gea.bau.edu.bd>
 */
public class TransferLearning extends AbstractClassifier {
    /**
     * For serialization.
     */
    private static final long serialVersionUID = -7891225050957072995L;

    /**
     * The final forest.
     */
    private Classifier forest;

    /**
     * Source Dataset from the knowledge is being transferred.
     */
    private Instances sourceDataset=null;
    
    /**
     * Target dataset.
     */
    private Instances targetDataset=null;
   /**
     * projected dataset.
     */
    private Instances projectedDataset=null;
   
    /**
     * The number of trees will be built for a forest. (default 2)
     */
    private int numTrees = 2;
//    /**
//     * The minimum number of records in a leaf for source forest. (default 20)
//     */
//    private int srcMinRecLeaf = 10;
    
//     /**
//     * The minimum number of records in a leaf for target forest. (default 10)
//     */
//    private int tgtMinRecLeaf = 10;

     /**
     * Used to regularize Ridge regression. (default 0.5)
     */
    private double sigma_ridge = 0.001;
    /**
     * Used to regularize MMD. (default 10)
     */
    private double lambda_mmd = 10.00;
    /**
     * Used to regularize manifold. (default 0.1)
     */
    private double gama_manifold = 0.100;

//    /**
//     * Used to determine mmd neighbours. (default 10)
//     */
//    private int nn_mmd = 10;
    boolean isMeanStd=true; //true-> mean+std, false->only mean
    boolean isDistinct=true;  //true-> remove identical class dist i.e. Opt 1, false->all, Opt 2
    boolean isOnlyContributoryAttribute=true; //true: only contributory attributes, false: all numerical attributes
    RulesToForest classifier=null;
    boolean printFinalClassifier=false;
    boolean printMsg=false;
    private int baseClassifier=0; //0: RF, 1:SysFor
    private double similarityThreshold=0.8;//used to find similar class distribution
    private String manifoldK_Range="";
    /**
     * A variable that is used to store the attribute domains of the passed
     * dataset.
     */

    /** the distance function used. */
    final String levelPadding="|   ";
    
    /**
     * Parses a given list of options. <br>
     *
     * <!-- options-start -->
     * Valid options are: <br>
     *
     * <pre> -N &lt;numTrees&gt;
     *  The number of trees to be built for a forest. (default 2)
     * </pre>
     * 
     * <pre> -L &lt;minSrcRecLeaf&gt;
     *  The minimum number of records in a source leaf. Works as in C4.5. (default 10)
     * </pre>
     *
     * <pre> -T &lt;minTgtRecLeaf&gt;
     *  The minimum number of records in a target leaf. Works as in C4.5. (default 10)</pre>
     *
     * <pre> -S &lt;Rigde regularizer&gt;
     *  The ridge regularizer to avoid overfitting to the training data. (default 0.5f)</pre>
     *
     * <pre> -G &lt;Manifold regularizer&gt;
     *  The Manifold regularizer to avoid overfitting to the training data. (default 0.1f)</pre>
     *
     * <pre> -M &lt;MMD regularizer&gt;
     *  The MMD regularizer to avoid overfitting to the training data. (default 10.0f)</pre>
     *
     * <!-- options-end -->
     *
     * Options after -- are passed to the designated classifier.<p>
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String gsNumtrees = Utils.getOption('N', options);
        if (gsNumtrees.length() != 0) {
            setNumTrees(Integer.parseInt(gsNumtrees));
        } else {
            setNumTrees(2);
        }

        String gRidge = Utils.getOption('S', options);
        if (gRidge.length() != 0) {
            setRidgeRegularizer(Float.parseFloat(gRidge));
        } else {
            setRidgeRegularizer(0.5f);
        }
        String gManifold = Utils.getOption('G', options);
        if (gManifold.length() != 0) {
            setManifoldRegularizer(Float.parseFloat(gManifold));
        } else {
            setManifoldRegularizer(0.1f);
        }
        String gMMD = Utils.getOption('M', options);
        if (gMMD.length() != 0) {
            setMMDRegularizer(Float.parseFloat(gMMD));
        } else {
            setMMDRegularizer(10.0f);
        }
        super.setOptions(options);
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return the current setting of the classifier
     */
    @Override
    public String[] getOptions() {

        Vector<String> result = new Vector<String>();
        
        result.add("-N");
        result.add("" + getNumTrees());

        result.add("-S");
        result.add("" + getRidgeRegularizer());

        result.add("-G");
        result.add("" + getManifoldRegularizer());

        result.add("-M");
        result.add("" + getMMDRegularizer());

        Collections.addAll(result, super.getOptions());

        return result.toArray(new String[result.size()]);
    }
    /**
     * Setter for Base classifier
     *
     * @param baseClassifier value 0: RF, 1: SysFor
     */
    public void setBaseClassifier(int baseClassifier)
    {
        this.baseClassifier=baseClassifier;
    }

    /**
     * Setter for ManifoldRegularizer
     *
     * @param gama_manifold value to set to
     */
    public void setManifoldRegularizer(double gama_manifold) {
        this.gama_manifold = gama_manifold;
    }

    /**
     * Getter for ManifoldRegularizer
     *
     * @return gama_manifold
     */
    public double getManifoldRegularizer() {
        return this.gama_manifold;
    }

    /**
     * Setter for MMDRegularizer
     *
     * @param lambda_mmd value to set to
     */
    public void setMMDRegularizer(float lambda_mmd) {
        this.lambda_mmd = lambda_mmd;
    }

    /**
     * Getter for MMDRegularizer
     *
     * @return lambda_mmd
     */
    public double getMMDRegularizer() {
        return this.lambda_mmd;
    }

    /**
     * Setter for RidgeRegularizer
     *
     * @param sigma_ridge value to set to
     */
    public void setRidgeRegularizer(double sigma_ridge) {
        this.sigma_ridge = sigma_ridge;
    }

    /**
     * Getter for RidgeRegularizer
     *
     * @return sigma_ridge
     */
    public double getRidgeRegularizer() {
        return this.sigma_ridge;
    }
    /**
     * Setter for forest size
     *
     * @param numTrees value to set to for forest size
     */
    public void setNumTrees(int numTrees) {
        this.numTrees = numTrees;
    }

    /**
     * Getter for forest size
     *
     * @return numTrees
     */
    public int getNumTrees() {
        return this.numTrees;
    }

    public String getManifoldKRange()
    {
        return this.manifoldK_Range;
    }
    
    /**
     * This method corresponds to build TLF.
     *
     * @param data - data with which to build the classifier
     * @throws java.lang.Exception
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        data = new Instances(data);
        targetDataset = data;
        if(sourceDataset==null) //no transfer learning, just build on target dataset
        {
            forest=TLUtils.buildClassifier(targetDataset, numTrees, TLUtils.findMinLeafSize(targetDataset.numInstances()), baseClassifier);
            String treeStr=TLUtils.preprocessTree(forest.toString());
            classifier=new RulesToForest(treeStr,targetDataset);
        }
        else
        {
             projectedDataset=projectedSourceData();
             if(projectedDataset==null) //no transfer learning, just build on target dataset
             {
                forest=TLUtils.buildClassifier(targetDataset, numTrees, TLUtils.findMinLeafSize(targetDataset.numInstances()), baseClassifier);
                String treeStr=TLUtils.preprocessTree(forest.toString());
                classifier=new RulesToForest(treeStr,targetDataset);
             }
             else{
                forest=TLUtils.buildClassifier(projectedDataset, numTrees, TLUtils.findMinLeafSize(projectedDataset.numInstances()), baseClassifier); 
                String treeStr=TLUtils.preprocessTree(forest.toString());
                classifier=new RulesToForest(treeStr,targetDataset);
             }
        }
        if(printFinalClassifier)
            System.out.println("\nTransfer learning classifier:\n"+forest.toString());
    }
    /**
   * Classifies the given test instance. The instance has to belong to a
   * dataset when it's being classified. Note that a classifier MUST
   * implement either this or distributionForInstance().
   *
   * @param instance the instance to be classified
   * @return the predicted most likely class for the instance or
   * Utils.missingValue() if no prediction is made
   * @exception Exception if an error occurred during the prediction
   */
    @Override
    public double classifyInstance(Instance instance) throws Exception{
      double pred=0.0;
      if(classifier==null)
      {
          throw new IllegalArgumentException(
            "TLF: Model is not built yet!!!");
      }
      else {
           pred=classifier.classifyInstance(instance);
        }
      return pred;
    }

    public void setSourceDataset(Instances data)throws Exception 
    {
        getCapabilities().testWithFail(data);
        data = new Instances(data);
        sourceDataset = data;
    }
    public Instances getSourceDataset()
    {
        return sourceDataset;
    }
    public Instances getProjectedDataset()
    {
        return projectedDataset;
    }
    @Override
    public String toString()
    {
        return forest.toString();
    }

    private Instances projectedSourceData()
    {
        Instances projectSourceTarget=null;
        Forest srcForest=new Forest(sourceDataset, numTrees);
        srcForest.buildForest();
        if(printMsg){
            TLUtils.display(srcForest.getClassDistribution(), "Source Class distribution");
            TLUtils.display(srcForest.getAttrContribution(), "Source Attribute contribution");
            TLUtils.display(srcForest.getLeafCentroids(), "Source Leaf Centroids");
        }
        Forest tgtForest=new Forest(targetDataset, numTrees);
        tgtForest.buildForest();
        if(printMsg){
            TLUtils.display(tgtForest.getClassDistribution(), "Target Class distribution");
            TLUtils.display(tgtForest.getAttrContribution(), "Target Attribute contribution");
            TLUtils.display(tgtForest.getLeafCentroids(), "Target Leaf Centroids");
        }
        //find similar class distributions
        double [][]JSD=TLUtils.calculateSimilarityMatrix(srcForest.getClassDistribution(),tgtForest.getClassDistribution(),
                sourceDataset,targetDataset);
        if(printMsg)TLUtils.display(JSD, "Similarity of class distributions");
        int [][]pivots=TLUtils.findPivots(JSD.clone(),similarityThreshold);
        if(pivots.length>0)
        {
        if(printMsg)TLUtils.display(pivots, "Pivots");
        double [][]ws=TLUtils.findAttributeContribution(srcForest.getAttrContribution(), pivots, 0);
        double [][]wt=TLUtils.findAttributeContribution(tgtForest.getAttrContribution(), pivots, 1);
        if(printMsg){
            TLUtils.display(ws, "source attribute contribution");
            TLUtils.display(wt, "target attribute contribution");
        }
        
        //find shared attribute contribution
        double [][]srcSAC=TLUtils.findAttributeContribution(srcForest.getLeafCentroids(), pivots, 0);
        double [][]tgtSAC=TLUtils.findAttributeContribution(tgtForest.getLeafCentroids(), pivots, 1);
        if(printMsg){
            TLUtils.display(srcSAC, "source shared attr cont");
            TLUtils.display(tgtSAC, "target shared attr cont");
        }
        //remove the class attribute
        int scol=srcSAC[0].length-1;
        double [][]X=new double[srcSAC.length][scol];
        for(int r=0;r<srcSAC.length;r++)
        {   System.arraycopy(srcSAC[r], 0, X[r], 0, scol);    
        }
        int tcol=tgtSAC[0].length-1;
        double [][]Y=new double[tgtSAC.length][tcol];
        for(int r=0;r<tgtSAC.length;r++)
        {   System.arraycopy(tgtSAC[r], 0, Y[r], 0, tcol);    
        }
        
        int [][]comAttr=findCommonAttributesAll();

        
        ArrayList<Integer> records=srcForest.getTransferrableInstances(pivots, 0);
        if(printMsg)System.out.println("\nTransferable source records\n"+records);
        Instances srcTrs=findTrasferableInstances(records);
        //merge and then normalize the source and target datasets with only common attributes
        Instances datasetComAttr=TLUtils.mergeInstancesTL(srcTrs,targetDataset);
        datasetComAttr.setClassIndex(datasetComAttr.numAttributes()-1);
        //manifold adaptation
        Matrix L=null;
        if(this.gama_manifold>0){
            double [][]k=TLUtils.calculateCosineSimilarity(datasetComAttr);
            Manifold mfold=new Manifold(datasetComAttr,k);
            L=mfold.getL();
            this.manifoldK_Range=mfold.getkRange();
            System.out.println("\nManifold: auto range of k values: "+this.manifoldK_Range+"\n");
            if(printMsg) System.out.println("\nManifold Graph Laplacian (L) matrix\n"+L);
        }

        //calculate projection matrix
        double [][]Ps=findCoefficients(srcTrs,targetDataset,X, Y, L.getArray());
        if(printMsg) TLUtils.displayProjection(Ps);  
        
        projectSourceTarget=projectSourceDataAndMergeTarget(Ps,srcTrs,comAttr);
        }
        if(printMsg) System.out.println("\nProjected source data\n"+projectSourceTarget);
        return projectSourceTarget;
    }
    private double[][] findCoefficients(Instances srcDS,Instances tgtDS,double [][]X, double[][]Y, double[][] L)
    {
        int row=srcDS.numAttributes()-1;
        int col=tgtDS.numAttributes()-1;
        double [][]Ps=new double[row][col];   
        double [][]p1=new double[row][col]; 
        double [][]p2=new double[row][col]; 
        for(int r=0;r<row;r++){
            Arrays.fill(Ps[r], 0);
            Arrays.fill(p1[r], 0);
            Arrays.fill(p2[r], 0);
        }
        String sf=".\\mfiles\\srctemp.arff";
        String tf=".\\mfiles\\tgttemp.arff";
        TLUtils.writeARRF(srcDS, sf);
        TLUtils.writeARRF(tgtDS, tf);
        try {
          System.out.println("Matlab started...");   
         //Start MATLAB asynchronously
            Future<MatlabEngine> eng = MatlabEngine.startMatlabAsync();
         // Get engine instance from the future result
            MatlabEngine ml = eng.get();
            ml.feval("addpath",".\\mfiles\\");
            System.out.println("Ridge started...");
            p1=(double[][])ml.feval("RidgeGea",X,Y,this.sigma_ridge);
            if(printMsg){TLUtils.display(p1, "P1");}
            System.out.println("Ridge ended...");
            int flag=1;
            if(this.lambda_mmd>0 ||this.gama_manifold>0){
            System.out.println("MMD started..");
            try{
                p2=(double[][])ml.feval("KernelAndMMD",sf,tf,L,this.sigma_ridge, this.lambda_mmd, this.gama_manifold);
            }
            catch(Exception e)
            {
                flag=0;
            }
            System.out.println("MMD ended...");
            if(printMsg){ TLUtils.display(p2, "p2");}
            }
            //merging p1 and p2
            for(int i=0;i<row;i++)
            {
                for(int j=0;j<col;j++)
                {
                    if((this.lambda_mmd>0 ||this.gama_manifold>0)&& flag==1)
                        Ps[i][j]=(p1[i][j]+p2[i][j])/2.0;
                    else
                        Ps[i][j]=p1[i][j];
                }
            }
            // Disconnect from the MATLAB session
            ml.disconnect();
            System.out.println("Matlab ended...");
        } catch (ExecutionException | InterruptedException e) {
            e.printStackTrace();
            Ps= TLUtils.ManifoldMMD(X, Y, this.sigma_ridge, this.gama_manifold);
        } 
        //finalising projection matrix
       finalProjection(Ps);
       return Ps;
    }
    private void finalProjection(double [][]Ps)
    {
        int row=Ps.length;
        int col=Ps[0].length;
        double diag=0.99;
        double rem=0;
        if (row>0)rem=(1.0-diag)/row;
        for(int j=0;j<col;j++)
        {           
            for(int i=0;i<row;i++)
            {    
                if(i==j)
                    Ps[i][j]=Ps[i][j]*diag;
                else    
                    Ps[i][j]=Ps[i][j]*rem;
                if(Ps[i][j]<0)Ps[i][j]=0;
            }
        }
    }
    private Instances findTrasferableInstances(ArrayList<Integer> records)
    {
        int s=records.size();
        Instances srcDS=new Instances(sourceDataset,0);
        //src data
        for(int i=0;i<s;i++)
        {
            int rno=records.get(i);
            srcDS.add(sourceDataset.get(rno));
        }
        srcDS.setClassIndex(-1);
        return srcDS;
    }
    

    private Instances projectSourceDataAndMergeTarget(double [][]Ps, Instances srcDS,int [][]comAttr)
    {
        String relation=targetDataset.relationName();
        int numAttr=targetDataset.numAttributes();
        ArrayList<Attribute> testAtts= new ArrayList<Attribute>();
        ArrayList<String> []testVals=new ArrayList[numAttr];
        for(int i=0;i<numAttr;i++)
        {
            if(targetDataset.attribute(i).isNominal())
            {
                testVals[i] = new ArrayList<String>();
                int numv=targetDataset.attribute(i).numValues();
                for(int k=0;k<numv;k++)
                    testVals[i].add(targetDataset.attribute(i).value(k));
            }
        }
        int p=Ps.length;//first value is the intercept
        int s=srcDS.numInstances();
        int t=targetDataset.numInstances();
        int total=s+t;
        double [][]alldata=new double[total][numAttr];  
        //src data
        for(int i=0;i<s;i++)
        {
            Instance inst=srcDS.get(i);
            for(int j=0;j<numAttr;j++)
            {
                if(targetDataset.attribute(j).isNominal())
                {
                    int sca=-1;
                    for(int m=0;m<comAttr.length;m++)
                    {
                        if(comAttr[m][1]==j)
                        {
                            sca=comAttr[m][0];
                        }
                    }
                    if(sca>=0){
                        double v=inst.value(sca);
                        String tv=srcDS.attribute(sca).value((int)v);
                        int ajIndex=testVals[j].indexOf(tv);
                        if(ajIndex>=0)
                            alldata[i][j]=ajIndex;
                        else
                        {
                            testVals[j].add(tv);
                            alldata[i][j]=testVals[j].indexOf(tv);
                        }
                    }
                }
                else
                {
                    double tv=0;
                    for(int c=0;c<p;c++)
                    {
                        tv+=inst.value(c)*Ps[c][j];
                    }
                    alldata[i][j]=tv;
                }                        
            }
        }
        //merging data
        for(int i=0,r=s;i<t && r<total;i++,r++)
        {
            Instance inst=targetDataset.get(i);
            for(int j=0;j<numAttr;j++)
            {
                if(targetDataset.attribute(j).isNominal())
                {
                    double v=inst.value(j);
                    String tv=targetDataset.attribute(j).value((int)v);
                    alldata[r][j]=testVals[j].indexOf(tv);
                }
                else
                {
                    alldata[r][j]=inst.value(j);
                }                        
            }
        }
        
        for(int i=0;i<numAttr;i++)
        {
            if(targetDataset.attribute(i).isNominal())
            {                
                testAtts.add(new Attribute(targetDataset.attribute(i).name(),testVals[i]));
            }
            else{
                testAtts.add(new Attribute(targetDataset.attribute(i).name()));
            }                        
        }
        
        Instances projectSourceTarget = new Instances(relation, testAtts, 0);
        for(int i=0;i<total;i++)
        {
            projectSourceTarget.add(new DenseInstance(1.0, alldata[i]));
        }
        projectSourceTarget.setClassIndex(targetDataset.classIndex());
        return projectSourceTarget;
    }
    private double[][] findBoundary()
    {
        int col=sourceDataset.numAttributes()-1;
        double [][]boundary=new double[4][col];//0: max, 1: min; 2: range, 3:weight
        double trange=0;
        for(int i=0;i<col;i++)
        {
            if(sourceDataset.attribute(i).isNumeric())
            {               
                boundary[0][i]=Math.max(sourceDataset.attributeStats(i).numericStats.max, 
                        targetDataset.attributeStats(i).numericStats.max);
                boundary[1][i]=Math.min(sourceDataset.attributeStats(i).numericStats.min, 
                        targetDataset.attributeStats(i).numericStats.min);
                 boundary[2][i] = boundary[0][i]-boundary[1][i];
                 trange+=boundary[2][i];
            }
        }
        if(trange>0){
        for(int i=0;i<col;i++)
        {
            if(sourceDataset.attribute(i).isNumeric())
            {
                boundary[3][i]=boundary[2][i]/trange;
            }
        }
        }
        return boundary;
    }
    private double[][] transformedAttrContribution(double [][]w, int nm, int flag)
    {
        int n=w.length;
        int noa=w[0].length;
        double[][] Q=new double[nm][noa];
        for(int i=0;i<nm;i++){
            if((flag==0 && i<n)||(flag==1 && i>=n)){
                int k=i;
                if(flag==1)k=i-n;
            for(int j=0;j<noa;j++){
                 Q[i][j]=w[k][j];
              }
            }
            else{
                Arrays.fill(Q[i], 0);
            }
        }
        return Q;
    }
    
    private int[][] findCommonAttributesNumeric()
    {
        int sa=sourceDataset.numAttributes();
        int ta=targetDataset.numAttributes();
        ArrayList<Integer>source = new ArrayList<Integer>();
        ArrayList<Integer>target = new ArrayList<Integer>();
        for(int i=0;i<sa;i++)
        {
            if(sourceDataset.attribute(i).isNumeric()){
                String aName=sourceDataset.attribute(i).name();
                for(int j=0;j<ta;j++) {
                    if(targetDataset.attribute(j).isNumeric()){
                        String tName=targetDataset.attribute(j).name();
                        if(aName.equals(tName)){
                            source.add(i);
                            target.add(j);
                            break;
                        }
                    }
                }
            }
        }
       int ss=source.size();
       int[][] comA=new int[ss][2]; 
       if(ss==target.size())
       {
           for(int i=0;i<ss;i++)
            {
                comA[i][0]=source.get(i);
                comA[i][1]=target.get(i);
            }
       }
       return comA.clone();
    }
    private int[][] findCommonAttributesAll()
    {
        int sa=sourceDataset.numAttributes();
        int ta=targetDataset.numAttributes();
        ArrayList<Integer>source = new ArrayList<Integer>();
        ArrayList<Integer>target = new ArrayList<Integer>();
        for(int i=0;i<sa;i++)
        {
            String aName=sourceDataset.attribute(i).name();
            for(int j=0;j<ta;j++){
                String tName=targetDataset.attribute(j).name();
                if(aName.equals(tName)){
                    source.add(i);
                    target.add(j);
                    break;
                }
            }            
        }
       int ss=source.size();
       int[][] comA=new int[ss][2]; 
       if(ss==target.size())
       {
           for(int i=0;i<ss;i++)
            {
                comA[i][0]=source.get(i);
                comA[i][1]=target.get(i);
            }
       }
       return comA.clone();
    }
    private class Manifold{
        Matrix L;
        Instances data;
        double[][] S;
        int knnMin;
        int knnMax;
        public Manifold(Instances data, double[][]S)
        {
          this.data=new Instances(data);
          this.S=S.clone();
          knnMin=Integer.MAX_VALUE;
          knnMax=Integer.MIN_VALUE;
          calculateL();
        }
        public String getkRange()
        {
            return knnMin+"-"+knnMax;
        }
        public Matrix getL()
        {
            return L;
        }
        private void calculateL()
        {
//            if(printMsg) System.out.println("\nthe values of k are:\n");
            double [][]ww=calculateW();
            double [][]dd=digD(ww);            
            double [][]ii=TLUtils.identityMatrix(data.numInstances());
            Matrix W=new Matrix(ww);
            Matrix D=new Matrix(dd);
            Matrix D_inverse=D.inverse();
            Matrix I=new Matrix(ii);
            Matrix tmp=D_inverse.times(W);
            Matrix tmp1=tmp.times(D_inverse);
            L=I.minus(tmp1);
        }
        
        private double[][] digD(double [][]W)
        {
            int nm=W.length;
            double [][]D=new double[nm][nm];
            for(int i=0;i<nm;i++)
            {
                double sum=0;
                for(int j=0;j<nm;j++)
                {
                    sum+=W[i][j];
                    D[i][j]=0;
                }
                D[i][i]=sum;
            }
            return D;
        }
        private double[][] calculateW()
        {
            int nm=data.numInstances();
            double [][]W=new double[nm][nm];
            for(int i=0;i<nm;i++)
            {
                for(int j=0;j<nm;j++)
                {
                    W[i][j]=calculateGraphAffinity(i,j);
                }
            }
            return W;
        }
        private double calculateGraphAffinity(int i,int j)
        {
            double w_ij=0;
            if(i==j)
                w_ij=S[i][j];
            else{
                ArrayList<Integer> knn_i=calculateGraphAffinity(i);
                ArrayList<Integer> knn_j=calculateGraphAffinity(j);                
                if(knn_i.contains(new Integer(j))||knn_j.contains(new Integer(i)))
                    w_ij=S[i][j];
            }
            return w_ij;
        }
        private ArrayList<Integer> calculateGraphAffinity(int i)
        {
            ArrayList<Integer> knn=new ArrayList<Integer>();
            int minK=4;
            int maxK=64;
            int nm=S.length;
            double [][]array=new double[nm][2];
            for(int k=0;k<nm;k++)
            {
                array[k][0]=S[k][i];
                array[k][1]=k;
            }
            Arrays.sort(array, new Comparator<double[]>() {
                public int compare(double[] a, double[] b) {
                    return Double.compare(a[0], b[0]);
                }
            });
            double cv=data.get(i).value(data.classIndex());            
            for(int k=nm-1;k>=0;k--)
            {
                int recNo=(int)array[k][1];
                if(recNo!=i)
                {
                    if((knn.size()<minK || cv==data.get(recNo).value(data.classIndex()))
                            && knn.size()<maxK )
                        knn.add(recNo);
                }
            }
//            if(printMsg) System.out.print(knn.size()+", ");
            if(knn.size()<knnMin)knnMin=knn.size();
            if(knn.size()>knnMax)knnMax=knn.size();
            return knn;
        }
    }
   
    
   private class MMD{
      Matrix M;
      double[][] mmd;
      Instances src;
      Instances tgt;
      double mu=0.5;
      public MMD(Instances src, Instances tgt)
      {
          this.src=new Instances(src);
          this.tgt=new Instances(tgt);
          calculateM();
      }
      public double[][] getMMD()
      {
          return mmd.clone();
      }
      public Matrix getM()
      {
          return M;
      }
      private void calculateM()
      {
          int n=src.numInstances();
          int m=tgt.numInstances();
          int nm=n+m;
          mmd=new double[nm][nm];
          for(int i=0;i<nm;i++)
          {
              Arrays.fill(mmd[i], 0);
          }
          ArrayList<String> commonCV=findCommonCV();
          int numc=commonCV.size();
          if(numc>0)
          {
          ArrayList<Integer>[] srcCWR=findClassWiseRecords(src,commonCV,0);
          ArrayList<Integer>[] tgtCWR=findClassWiseRecords(tgt,commonCV,n);
          estimate_mu(srcCWR,tgtCWR,commonCV,n);
          if(printMsg) System.out.println("\nMMD mu:"+mu);
          
          //find marginal distribution
          double[][] m_m=new double[nm][nm];
          double sn=1.0/(n*n);
          double tm=1.0/(m*m);
          double stmn=-1.0/(n*m);
          for(int i=0;i<nm;i++)
          {
              for(int j=0;j<nm;j++)
              {
                  if(i<n && j<n)
                      m_m[i][j]=sn;
                  else if(i>=n && j>=n)
                      m_m[i][j]=tm;
                  else
                      m_m[i][j]=stmn;
              }
          }
          double[][] m_c=new double[nm][nm];
          for(int i=0;i<nm;i++)
          {
              Arrays.fill(m_c[i], 0);
          }
          
          for(int c=0;c<numc;c++)
          {
             int nc=srcCWR[c].size();
             int mc=tgtCWR[c].size();
             sn=1.0/(nc*nc);
             tm=1.0/(mc*mc);
             stmn=-1.0/(nc*mc);
             for(int i=0;i<nm;i++)
                {
                    for(int j=0;j<nm;j++)
                    {
                        if(srcCWR[c].contains(new Integer(i)) && srcCWR[c].contains(new Integer(j)))
                            m_c[i][j]+=sn;
                        else if(tgtCWR[c].contains(new Integer(i)) && tgtCWR[c].contains(new Integer(j)))
                            m_c[i][j]+=tm;
                        else if((srcCWR[c].contains(new Integer(i)) && tgtCWR[c].contains(new Integer(j)))
                                ||(tgtCWR[c].contains(new Integer(i)) && srcCWR[c].contains(new Integer(j))))
                            m_c[i][j]+=stmn;
                    }
                }
          }
 
          for(int i=0;i<nm;i++)
          {
              for(int j=0;j<nm;j++)
              {
                  mmd[i][j]=(1-mu)*m_m[i][j]+mu*m_c[i][j];
              }
          }
          }
          M=new Matrix(mmd);
      }
      private void estimate_mu(ArrayList<Integer>[] srcCWR,ArrayList<Integer>[] tgtCWR,
              ArrayList<String> commonCV, int start)
      {
          double adist_m = proxy_a_distance(src, tgt);
          Instances srctmp;
          Instances tgttmp;
          int numc=commonCV.size();
          double terror=0;
          for (int i = 0; i < numc; i++) { 
              srctmp=new Instances(src,0);
              int nr=srcCWR[i].size();
              for(int j=0;j<nr;j++){
                  srctmp.add(src.instance(srcCWR[i].get(j)));
              }                  
              tgttmp=new Instances(tgt,0);
              nr=tgtCWR[i].size();
              for(int j=0;j<nr;j++){
                  tgttmp.add(tgt.instance((tgtCWR[i].get(j)-start)));
              }
              if(srctmp.numInstances()>0 && tgttmp.numInstances()>0){
              double adist_tmp = proxy_a_distance(srctmp, tgttmp);
              terror+=adist_tmp;
              }
          }
          double adist_c=terror/numc;
          if((adist_c+adist_m)<=0)
          {
              mu=0.5;
          }
          else
          {
            mu=adist_c/(adist_c+adist_m);
            if(mu>1)
                mu=1;
            else if(mu<0)
                mu=0;
          }
      }
      private ArrayList<Integer>[] findClassWiseRecords(Instances data,ArrayList<String> commonCV, int start)
      {
          int numc=commonCV.size();
          ArrayList<Integer>[]cvRecords=new ArrayList[numc];
          for (int i = 0; i < numc; i++) { 
                   cvRecords[i] = new ArrayList<Integer>(); 
          }
          int ci=data.classIndex();
          int n=data.numInstances();
          for (int i = 0; i < n; i++) { 
              Instance inst=data.get(i);
              String v=data.attribute(ci).value((int)inst.value(ci));
              int index=commonCV.indexOf(v);
              if(index>=0)
              {
                  cvRecords[index].add(i+start);
              }
          }
          return cvRecords;
      }
      
      private ArrayList<String> findCommonCV()
      {
          int ci=src.classIndex();
          ArrayList<Double> sv=new ArrayList<Double>();
          int ns=src.numInstances();
          for(int i=0;i<ns;i++)
          {
              double d=src.get(i).value(ci);
              if(!sv.contains(d))
              {
                  sv.add(d);
              }
          }
          ArrayList<Double> tv=new ArrayList<Double>();
          int ts=tgt.numInstances();
          for(int i=0;i<ts;i++)
          {
              double d=tgt.get(i).value(ci);
              if(!tv.contains(d))
              {
                  tv.add(d);
              }
          }
          ArrayList<String> ccv=new ArrayList<String>();
          ArrayList<String> scv=new ArrayList<String>();
          int numCV=sv.size();
          for(int k=0;k<numCV;k++)
          {
            scv.add(src.attribute(ci).value(sv.get(k).intValue()));
          }
          numCV=tv.size();
          for(int k=0;k<numCV;k++)
          {
              String cv=tgt.attribute(ci).value(tv.get(k).intValue());
              if(scv.contains(cv))  {
                ccv.add(cv);
            }
          }
          return ccv;
      }
      private double proxy_a_distance(Instances srcData, Instances tgtData)
      {
          double dist=0;
          Instances srcCc=TLUtils.changeClassValue(srcData, "S");
          Instances tgtCc=TLUtils.changeClassValue(tgtData, "T");
          Instances mergeInsts=TLUtils.mergeInstancesTL(srcCc, tgtCc);
          mergeInsts.setClassIndex(src.classIndex());
          Classifier model=TLUtils.buildClassifier(mergeInsts, 1, TLUtils.findMinLeafSize(mergeInsts.numInstances()), baseClassifier);
          double error=1-TLUtils.calculateAccuracy(model, mergeInsts);
          dist= 2 * (1 - 2 * error);
          return dist;
      }
   } 
    
    
   private class Forest {
    List<Node> allLeaves;   
    Instances dataset;   
    int numberTrees;
    int minLeafSize;
    int classIndex;
    String cv[];
    double [][]attributeContribution;
    double [][]classdistribution;
    double [][]leafCentroids;
    public Forest(Instances dataset,int numberTrees)
    {
        this.dataset=dataset;
        this.minLeafSize=TLUtils.findMinLeafSize(dataset.numInstances());
        this.numberTrees=numberTrees;
    }
    public void buildForest()
    {
        Classifier model=TLUtils.buildClassifier(dataset,numberTrees,minLeafSize,baseClassifier);
        String treeStr=model.toString();
        if(baseClassifier==0) treeStr=TLUtils.preprocessTree(treeStr);
        if(printMsg) System.out.println(treeStr);
        RulesToForest rtf=new RulesToForest(treeStr,dataset);
        rtf.assignRecords();
        generateAttributeContributionAndClassDist(rtf);  
    }
    
    public String[]getClassValues()
    {
        return this.cv.clone();
    }
    public double[][]getLeafCentroids()
    {
        return this.leafCentroids.clone();
    }
    public double[][]getAttrContribution()
    {
        return this.attributeContribution.clone();
    }
    public double[][]getClassDistribution()
    {
        return this.classdistribution.clone();
    }
    public ArrayList<Integer> getTransferrableInstances(int [][]pivots, int index)
    {
        ArrayList<Integer> records=new ArrayList<Integer>();
        for(int i=0;i<pivots.length;i++)
        {
            int lno=pivots[i][index];
            Node leaf=allLeaves.get(lno);
            List<Integer> insts=leaf.getLeafData();
            for(int r=0;r<insts.size();r++)
            {
               if(!records.contains(new Integer(insts.get(r))))
               {
                   records.add(insts.get(r));
               }
            }
        }
        return records;
    }
    private void generateAttributeContributionAndClassDist(RulesToForest rtf)
    {
        allLeaves=rtf.getTotalLeaves();
        int numLeaves=allLeaves.size();
        int numAttr=dataset.numAttributes();
        int numCV=dataset.numClasses();
        cv=new String[numCV];
        for(int i=0;i<numCV;i++)
        {
            cv[i]=dataset.classAttribute().value(i);
        }
        classdistribution=new double[numLeaves][numCV];
        attributeContribution=new double[numLeaves][numAttr];
        leafCentroids=new double[numLeaves][numAttr];
        double[][] nominalDists = new double[numAttr][];
        for(int i=0;i<numLeaves;i++)
        {
            Node leaf=allLeaves.get(i);
            
            //find class distribution, attribute contribution (mean, std)
            //initialize nominal distribution
            for (int j = 0; j < numAttr; j++) {
                if (dataset.attribute(j).isNominal()) {
                    if(i==0)
                    nominalDists[j] = new double[dataset.attribute(j).numValues()];
                    Arrays.fill(nominalDists[j], 0.0);
                }
             } 
            //calculate total and nominal distribution
            List<Integer> records=leaf.getLeafData();
            if(records.size()>0){            
            for (int j = 0; j < numAttr; j++) {                    
                double sum=0, avg=0;  
                double []num=new double[records.size()];
                for(int k=0;k<records.size();k++)
                {
                    Instance inst=dataset.get(records.get(k));
                    if (dataset.attribute(j).isNominal()) {
                        nominalDists[j][(int)inst.value(j)] += inst.weight();
                    }
                    else{
                        num[k]=inst.value(j);
                        sum+=inst.value(j);
                    }                    
                }
                if(j!=dataset.classIndex()){
                    if (dataset.attribute(j).isNominal()) {
                        double max = -Double.MAX_VALUE;
                        int maxIndex = -1;
                        for (int d = 0; d < nominalDists[j].length; d++) {
                          if (nominalDists[j][d] > max) {
                            max = nominalDists[j][d];
                            maxIndex = d;
                          }
                        }
                        if(maxIndex>=0)
                        leafCentroids[i][j]=maxIndex;//dataset.attribute(j).value(maxIndex);
                        attributeContribution[i][j]=0;
                    }
                    else{
                        avg=(double)sum/records.size();
                        double std=calculateStd(num.clone(),avg);
                        if(isMeanStd && std>0)
                            avg=avg+Math.log(std)/Math.log(2.0);
//                            avg=avg+Math.abs(Math.log(std)/Math.log(2.0));
                        leafCentroids[i][j]=avg;   
                        attributeContribution[i][j]=avg;
                    }
                }
                else if(j==dataset.classIndex()){
                        leafCentroids[i][j]=dataset.attribute(j).indexOfValue(leaf.getLeafPrediction());
                        attributeContribution[i][j]=0;
                        for (int d = 0; d < nominalDists[j].length; d++) {
                            classdistribution[i][d] = nominalDists[j][d]/records.size();
                          }
                }
            }
            //find the attributes that have contribution for this leaf
            if(isOnlyContributoryAttribute){
                List<String> testAttr=leaf.getTestAttributes();
                for(int j=0;j<numAttr;j++){
                    if(!testAttr.contains(dataset.attribute(j).name()))
                       attributeContribution[i][j]=0;
                }
            }
            }
        }
       //remove identical class distribution
        if(isDistinct){
          int []flag=removeIdenticalDistribution(attributeContribution,classdistribution,leafCentroids);
          int n=flag.length;
          //merge identical leaves to a single one 
          for(int i=n-1;i>=0;i--){
              if(flag[i]>=0){
                  int targetIndex=flag[i];
                  Node leaf=allLeaves.get(i);
                  List<Integer> recNo=leaf.getLeafData();
                  for(int j=0;j<recNo.size();j++){
                      allLeaves.get(targetIndex).addInstance(recNo.get(j));
                  }
              }
          }
          //remove identical leaves
          for(int i=n-1;i>=0;i--){
              if(flag[i]>=0){
                  allLeaves.remove(new Integer(flag[i]));                 
              }
          }
        }
    }


    private double calculateStd(double []numbers, double average)
    {
        double std=0.0;
        for (int i=0; i<numbers.length;i++)
        {
            std += ((numbers[i] - average)*(numbers[i] - average)) / (numbers.length - 1);
        }            
        return Math.sqrt(std);
    }
    


    private int [] removeIdenticalDistribution(double [][]attrData,double [][]classData,double [][]leafMean)
    {
        int len=attrData.length;
        int acol=attrData[0].length;
        int ccol=classData[0].length;
        int []flag=new int [len];
        Arrays.fill(flag, -1);// all leaves are considered as distinct
        for(int l=0;l<len-1;l++){
            if(flag[l]==-1){
                for(int p=l+1;p<len;p++){
                    if(flag[p]==-1){
                        if(isIdentical(classData[l],classData[p])){
                           flag[p]=l;  //current leaf is identical to the l'th leaf.
                        }
                    }
                }
            }
        }
        int newlen=0;
        for(int l=0;l<len;l++){
            if(flag[l]==-1){
                newlen++;
            }
         }
        if(len>newlen){
            
            int[][] nominalDists = new int[acol][];
            for (int j = 0; j < acol; j++) {
                if (dataset.attribute(j).isNominal()) {
                    int ds=dataset.attribute(j).numValues();
                    nominalDists[j] = new int[ds];
                }
             }
            double [][]newAttrData=new double[newlen][acol];
            double [][]newClassData=new double[newlen][ccol];
            double [][]newLeafMean=new double[newlen][acol];
            for(int l=0,n=0;l<len && n<newlen;l++){
                if(flag[l]==-1){
                    System.arraycopy(classData[l], 0, newClassData[n], 0, ccol);
                    int t=1;
                    double []avg=new double[acol];
                    double []avgc=new double[acol];
                    for (int j = 0; j < acol; j++) {
                        
                        if (dataset.attribute(j).isNominal()) {
                            Arrays.fill(nominalDists[j], 0);
                             nominalDists[j][(int)leafMean[l][j]]++;
                        }
                        else{
                            avg[j]=attrData[l][j];
                            avgc[j]=leafMean[l][j];
                        }
                     }
                    for(int p=l+1;p<len;p++)
                    {
                        if(flag[p]==l){
                           for(int g=0;g<acol;g++){
                               if (dataset.attribute(g).isNominal()) {
                                   nominalDists[g][(int)leafMean[l][g]]++;
                               }
                               else{
                                    avg[g]=avg[g]+attrData[p][g];
                                    avgc[g]=avgc[g]+leafMean[p][g];
                               }}
                           t++;
                        }                        
                    }
                    for(int g=0;g<acol;g++)
                    {
                        if (dataset.attribute(g).isNominal()) {
                            int ds=nominalDists[g].length;
                            int mx=0;
                            for(int m=1;m<ds;m++){
                                if(nominalDists[g][m]>nominalDists[g][mx]){
                                    mx=m;
                                }
                            }
                            newLeafMean[n][g]=mx;//nominalDV[g][mx];
                        }
                        else{
                             newAttrData[n][g]=avg[g]/t;
                             newLeafMean[n][g]=avgc[g]/t;
                        }
                    }
                   n++;
                }
             }
            this.leafCentroids=newLeafMean.clone();
            this.attributeContribution=newAttrData.clone();
            this.classdistribution=newClassData.clone();
        }
        return flag.clone();
    }

    private boolean isIdentical(double []x, double []y)
    {
        boolean f=true;
        for(int i=0;i<x.length;i++)
        {
            if(x[i]!=y[i])
            {
                f=false;break;
            }
        }
        return f;
    }
    


    }  
    
    
    
}
