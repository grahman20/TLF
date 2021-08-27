function D = ReadArffNew(fName)
    %## read file
    loader = weka.core.converters.ArffLoader();
    loader.setFile( java.io.File(fName) );
    D = loader.getDataSet();
end

