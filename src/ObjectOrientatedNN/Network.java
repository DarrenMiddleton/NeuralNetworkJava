package ObjectOrientatedNN;

import java.util.Arrays;

import CrossNetworkFeatures.DataHandler;

public class Network {

	private boolean dropOut;
	private boolean dropConnect;
    private double dropOutRatio;
    private double dropConnectRatio;
	
    private Layer[] layers;
    private Layer inputLayer;
    private Layer outputLayer;
    private String networkID;
    private String activationFunction = "TanH";

    //used to give feedback on neural network 
    int modulusNumber;
     
    /**
     * Initializes a neural network
     * @param networkID name of neural network
     * @param sizes length of arrays = number of neurons and each index is that layers number of neurons
     */
    public Network(String networkID, int... sizes) {
    	this.modulusNumber = Integer.MAX_VALUE;
    	this.dropConnect = false;
    	this.dropOut = false;
    	this.networkID = networkID;
        this.dropOutRatio = 0;
        
        layers = new Layer[sizes.length];
        for(int i = 0; i < sizes.length; i++) {
            Layer layer;
            if(i == 0) {
                layer = new Layer(null, sizes[i]);
            }else {
                layer = new Layer(layers[i-1],sizes[i]);
                layer.createWeightsAndBias();
            }
            layers[i] = layer;
        }
        inputLayer = layers[0];
        outputLayer = layers[sizes.length-1];
    }

    public double[] calculate(double... in) {
    	inputLayer.setOutput(in);
        for(int i = 1; i < layers.length; i++) {
            if(i == layers.length-1) {
                layers[i].calculate("Sigmoid");
            } else {
                layers[i].calculate(activationFunction);
            }
        }
        return outputLayer.getOutput();
    }

    public double train(double[] in, double[] exp, double eta) {

    	if(dropOut){
        	for(int currentLayer = 1; currentLayer < layers.length-1; currentLayer++){
        		for(int n = 0; n < layers[currentLayer].neurons.length; n++){
        			if (Math.random() < dropOutRatio){
        				layers[currentLayer].neurons[n].setDropout(true);
        				}
        			}
        	}
    	}
    	
    	if(dropConnect){
        	for(int currentLayer = 1; currentLayer < layers.length-1; currentLayer++){
        		for(int n = 0; n < layers[currentLayer].neurons.length; n++){
        			if (Math.random() < dropConnectRatio){
        				layers[currentLayer].neurons[n].setDropout(true);
        				}
        			}
        	}
    	}

        calculate(in);
        outputLayer.calculateOutputErrorDerivative(exp);
        for(int i = layers.length-2; i > 0; i--) {
            layers[i].backwardspropagation();
        }
        for(int i = 1; i < layers.length; i++) {
            layers[i].updateWeights(eta);
        }
        double sum = 0;
        for (int i = 0; i< outputLayer.getNeurons().length; i++){
            sum =+ (outputLayer.getNeurons()[i].getOutputValue() - exp[i]) *
                    (outputLayer.getNeurons()[i].getOutputValue() - exp[i]);
        }
        if(dropOut){
        	for(int i = 1; i < layers.length-1; i++){
        		for(int n = 0; n < layers[i].neurons.length; n++){
    				layers[i].neurons[n].setDropout(false);
    			}
    		}
        }
        if(dropConnect){
        	for(int i = 1; i < layers.length-1; i++){
        		for(int n = 0; n < layers[i].neurons.length; n++){
    				layers[i].neurons[n].setDropout(false);
    			}
    		}
        }
        
        return sum/outputLayer.getNeurons().length;
    }

    public void train(DataHandler trainSet, int iterations, int batch_size, double eta) {

        for(int currentIteration = 0; currentIteration < iterations; currentIteration++) {

            double totalError = 0;
            DataHandler batch = trainSet.getNewBatch(batch_size);
            for(int k = 0; k < batch.size(); k++) {
                double a = this.train(batch.getInput(k), batch.getOutput(k), eta);
                totalError+= a;

                if (currentIteration % modulusNumber == 0 ){
                	System.out.println("Mean squared error of current data = "  + a + " Expected Out         =" + Arrays.toString(batch.getOutput(k)) +   "    calculated         = "+ Arrays.toString(this.outputLayer.getOutput()));
                }
            }
        }
    }

    public void debugNetwork(boolean derivatives, boolean error_signals, boolean weights) {
        System.out.println("###########################################################");
        for(Layer l:layers) {
        	System.out.println("Layer size: " + l.neurons_amount);
        	System.out.println("Output: "+Arrays.toString(l.getOutput()));
            if(derivatives){
                double[] der = new double[l.neurons_amount];
                for(int i = 0; i < l.neurons_amount; i++) {
                    der[i] = l.getNeurons()[i].getOutputDerivative();
                }
                System.out.println("Derivatives: "+Arrays.toString(der));

            }if(error_signals){
                double[] der = new double[l.neurons_amount];
                for(int i = 0; i < l.neurons_amount; i++) {
                    der[i] = l.getNeurons()[i].getErrorValue();
                }
                System.out.println("Error values: "+Arrays.toString(der));

            }
        }
        System.out.println("###########################################################");
    }

    public void setDropConnectRatio(double dropConnectRatio){
    	this.dropConnectRatio = dropConnectRatio;
    }
    
    public void setDropoutRatio(double dropoutRatio){
    	this.dropOutRatio = dropoutRatio;
    }
    
    public void setDropConnect(boolean dropConnect){
    	this.dropConnect = dropConnect;
    }
    
    public void setDropOut(boolean dropout){
    	this.dropOutRatio = dropOutRatio;
    }
    
    public void setActivationTanh(){
    	this.activationFunction = "TanH";
    }
    
    public void setActivationSigmoid(){
    	this.activationFunction = "Sigmoid";
    }
    
    public void setActivationLeakyRelu(){
    	this.activationFunction = "LeakyRelu";
    }
    
    public void setModulusNumber(int newModulusNumber){
    	this.modulusNumber = newModulusNumber;
    }
}

