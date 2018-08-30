package NeuralNetwork;

import java.util.Arrays;

import CrossNetworkFeatures.DataHandler;

public class NeuralNetwork {

    private double[][] output;
    private double[][][] weights;
    private double[][] bias;
    private double[][] errorValue;
    private double[][] outputDerivative;
    private boolean dropOut;
    private boolean dropConnect;
    private final int[] networkLayers;
    private final int   inputSize;
    private final int   outputSize;
   
    public NeuralNetwork(int... networkLayers) {
    	this.dropConnect = false;
    	this.dropOut = true;
        this.networkLayers = networkLayers;
        this.inputSize = networkLayers[0];
        this.outputSize = networkLayers[networkLayers.length-1];

        this.output = new double[networkLayers.length][];
        this.weights = new double[networkLayers.length][][];
        this.bias = new double[networkLayers.length][];
        this.errorValue = new double[networkLayers.length][];
        this.outputDerivative = new double[networkLayers.length][];

        for(int i = 0; i < networkLayers.length; i++) {
            this.output[i] = new double[networkLayers[i]];
            this.errorValue[i] = new double[networkLayers[i]];
            this.outputDerivative[i] = new double[networkLayers[i]];
            this.bias[i] = new double[networkLayers[i]];
            
            for(int biasindex = 0; biasindex < networkLayers[i]; biasindex++){
            	bias[i][biasindex] = (1/Math.sqrt(networkLayers[i]) *  (2 * Math.random() - 1));   
            }
            
            if(i > 0) {
                weights[i] = new double[networkLayers[i]][networkLayers[i-1]];
                for(int prevLayer = 0; prevLayer < networkLayers[i-1]; prevLayer++){
                	for(int currentLayer = 0; currentLayer < networkLayers[i]; currentLayer++){
                		weights[i][currentLayer][prevLayer] = (1/Math.sqrt(networkLayers[i]) *  (2 * Math.random() - 1));
                	}
                }
            }
        }
    }

    public double[] calculate(double... input) {
        if(input.length != this.inputSize) return null;
        this.output[0] = input;
        for(int layer = 1; layer < networkLayers.length; layer ++) {
            for(int neuron = 0; neuron < networkLayers[layer]; neuron ++) {
                double sum = bias[layer][neuron];
                for(int prevNeuron = 0; prevNeuron < networkLayers[layer-1]; prevNeuron ++) {
                    sum += output[layer-1][prevNeuron] * weights[layer][neuron][prevNeuron];
                }
                output[layer][neuron] = sigmoid(sum);
                outputDerivative[layer][neuron] = output[layer][neuron] * (1 - output[layer][neuron]);
            }
        }
        double[] newOutputArray = new double[outputSize];
        for(int i = 0; i < outputSize; i++){
        	newOutputArray[i] = output[networkLayers.length-1][i];
        	}
        return output[networkLayers.length-1];
    }

    public void train(DataHandler dataSet, int loops, int batchSize) {
        for(int i = 0; i < loops; i++) {
            DataHandler batch = dataSet.extractDataSet(batchSize);
            for(int b = 0; b < batchSize; b++) {
                this.train(batch.getInput(b), batch.getOutput(b), 0.3);
            }
//            System.out.println(MSE(batch));
        }
    }

    public double MSE(double[] input, double[] target) {
    	if(checkInputOutputSize(input.length, target.length)){
    		calculate(input);
    		double v = 0;
    		for(int i = 0; i < target.length; i++) {
    			v += (target[i] - output[networkLayers.length-1][i]) * (target[i] - output[networkLayers.length-1][i]);
    			}
    		return v / (2d * target.length);
    		} else {
    			return 0;
    	}
    }

    public double MSE(DataHandler set) {
        double v = 0;
        for(int i = 0; i< set.size(); i++) {
            v += MSE(set.getInput(i), set.getOutput(i));
        }
        return v / set.size();
    }

    public void train(double[] input, double[] target, double eta) {
    	if(checkInputOutputSize(input.length, target.length)){
    		calculate(input);
    		backwardsProp(target);
    		updateWeights(eta);
    	}
    }

    public void backwardsProp(double[] target) {
        for(int neuron = 0; neuron < networkLayers[networkLayers.length-1]; neuron ++) {
            errorValue[networkLayers.length-1][neuron] = (output[networkLayers.length-1][neuron] - target[neuron])
                    * outputDerivative[networkLayers.length-1][neuron];
        }
        for(int layer = networkLayers.length-2; layer > 0; layer --) {
            for(int neuron = 0; neuron < networkLayers[layer]; neuron ++){
                double sum = 0;
                for(int nextNeuron = 0; nextNeuron < networkLayers[layer+1]; nextNeuron ++) {
                    sum += weights[layer + 1][nextNeuron][neuron] * errorValue[layer + 1][nextNeuron];
                }
                this.errorValue[layer][neuron] = sum * outputDerivative[layer][neuron];
            }
        }
    }

    public void updateWeights(double eta) {
        for(int layer = 1; layer < networkLayers.length; layer++) {
            for(int neuron = 0; neuron < networkLayers[layer]; neuron++) {

                double delta = - eta * errorValue[layer][neuron];
                bias[layer][neuron] += delta;

                for(int prevNeuron = 0; prevNeuron < networkLayers[layer-1]; prevNeuron ++) {
                    weights[layer][neuron][prevNeuron] += delta * output[layer-1][prevNeuron];
                }
            }
        }
    }
    
    private boolean checkInputOutputSize(int inputSize, int outputSize){
        if(inputSize != this.inputSize || outputSize != this.outputSize){
        	return false;
        }
        return true;
    }

    public double tanh(double x){
        return Math.tanh(x);
    }

    public double derivativeTanh(double x){
        return 1 - (tanh(x) * tanh(x));
    }

    public double sigmoid(double x) {
        return 1d / (1 + Math.exp(-x));
    }

    public double derivativeSigmoid(double x) {
        return sigmoid(x) * (1- sigmoid(x));
    }

    public double leakyRelu(double x) {return x > 0 ? x : x *0.1;}
    public double derivativeLeakyRelu(double x) { return x > 0 ? 1 : 0.1;}


}

