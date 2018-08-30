package ObjectOrientatedNN;

public class Neuron {

    private double[] weights;
    private double bias;
    private int neuronIndex;

    private double errorValue;
    private double outputValue;
    private double outputDerivative;
    private boolean dropout;
    private boolean dropConnect;

    public Neuron(int neuron_index) {
        this.neuronIndex = neuron_index;
    }

    public void initWeightsAndBias(int prevNeurons, double lowerW, double upperW, double lowerB, double upperB) {
        weights = new double[prevNeurons];
        bias = Math.random() * (upperB - lowerB) + lowerB;
        for(int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() * (upperW - lowerW) + lowerW;
        }
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

    public void calculate(Layer prevLayer, String function) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            if (!prevLayer.getNeurons()[i].dropout|| !prevLayer.getNeurons()[i].dropConnect) {
                sum += prevLayer.getNeurons()[i].getOutputValue() * weights[i];
            }
                if(function.equals("Tanh")) {
                    outputValue = tanh(sum);
                    outputDerivative = derivativeTanh(sum);
                } else if(function.equals("LeakyRelu")){
                	outputValue = leakyRelu(sum);
                	outputDerivative = derivativeLeakyRelu(sum);
                } else {
                    outputValue = sigmoid(sum);
                    outputDerivative = derivativeSigmoid(sum);
                }
           }
    }
    public void backprop(Layer nextLayer) {
        double sum = 0;
        for(int i = 0; i < nextLayer.getNeurons().length; i++){
        	if(!nextLayer.neurons[i].dropout) {
            sum += nextLayer.getNeurons()[i].getWeights()[neuronIndex] * nextLayer.getNeurons()[i].getErrorValue();
        	}
        }
        errorValue = outputDerivative * sum;

    }

    public void updateWeights(Layer prevLayer, double eta) {

        double delta = - eta * errorValue;
        bias += delta;
        for(int i = 0; i < prevLayer.neurons_amount;i++) {
        	if(!prevLayer.neurons[i].dropout) {
            weights[i] += delta * prevLayer.getNeurons()[i].getOutputValue();
        	}
        }
    }

    public double getErrorValue() {
        return errorValue;
    }

    public double getBias() {
        return bias;
    }

    public double getOutputDerivative() {
        return outputDerivative;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public double[] getWeights() {
        return weights;
    }

    public int getNeuronIndex() {
        return neuronIndex;
    }
    
    public boolean getDropout(){
    	return dropout;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setDropout(boolean dropout) {
        this.dropout = dropout;
    }

    public void setErrorValue(double errorValue) {
        this.errorValue = errorValue;
    }

    public void setNeuronIndex(int neuronIndex) {
        this.neuronIndex = neuronIndex;
    }

    public void setOutputDerivative(double outputDerivative) {
        this.outputDerivative = outputDerivative;
    }

    public void setOutputValue(double outputValue) {
        this.outputValue = outputValue;
    }

    public void loadFromString(String s) {
        String[] ar = s.split(",");
        for(int i = 0; i < weights.length; i++){
            weights[i] = Double.parseDouble(ar[i]);
        }
        bias = Double.parseDouble(ar[ar.length-1]);
    }

}
