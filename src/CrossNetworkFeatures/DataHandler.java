package CrossNetworkFeatures;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;

public class DataHandler {
    private int[] batchNumbers;

    private ArrayList<double[][]> inputAndOutput = new ArrayList<>();

    public DataHandler getNewBatch(int size) {
        if(size > 0 && size <= this.size()) {
            DataHandler newData = new DataHandler();
            batchNumbers = new int[size];

            for (int i = 0; i < size; i++ ){
                batchNumbers[i] = ThreadLocalRandom.current().nextInt(0, size + 1);
            }
            for(int i = 0 ; i < batchNumbers.length ; i++) {
                newData.insertData(this.getInput(i),this.getOutput(i));
            }
            return newData;
        }else return this;
    }

    public void insertData(double[]input, double[] output){
        inputAndOutput.add(new double[][] {input, output});
    }

    public void generateLogs(){
        for (int i = 0; i < inputAndOutput.size(); i++){
            System.out.println(Arrays.toString(inputAndOutput.get(i)));
        }
    }

    public DataHandler extractDataSet(int size){
        DataHandler dataHandler = new DataHandler();
        for (int i = 0; i <size; i++){
            dataHandler.insertData(this.getInput(i), this.getOutput(i));
        }
        return dataHandler;
    }

    public double[] getOutput(int index){
        return inputAndOutput.get(index)[1];
    }

    public int size() {
        return inputAndOutput.size();
    }

    public double[] getInput(int index){
        return inputAndOutput.get(index)[0];
    }
}
