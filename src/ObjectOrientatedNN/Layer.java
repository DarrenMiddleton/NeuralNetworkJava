package ObjectOrientatedNN;

public class Layer {

    int neurons_amount;
    Neuron[] neurons;

    Layer previousLayer;
    Layer nextLayer;
    boolean dropOutlayer;

    public Layer(Layer prevLayer, int neurons_amount) {
        this.dropOutlayer = dropOutlayer;
        this.neurons_amount = neurons_amount;
        this.neurons = new Neuron[neurons_amount];
        this.previousLayer = prevLayer;
        if(this.previousLayer != null) {
            this.previousLayer.nextLayer = this;
        }

        for(int i = 0; i < neurons_amount; i++) {
            neurons[i] = new Neuron(i);
        }
    }

    void createWeightsAndBias() {
        for(Neuron n:neurons) {
            n.initWeightsAndBias(previousLayer.neurons_amount, -0.3,0.3,-0.3,0.3);
        }
    }

    public void setOutput(double... values) {
        for(int i = 0; i < neurons_amount; i++) {
            neurons[i].setOutputValue(values[i]);
        }
    }

    public double[] getOutput() {
        double[] out = new double[neurons_amount];
        for(int i = 0; i < neurons_amount; i++) {
            out[i] = neurons[i].getOutputValue();
        }
        return out;
    }

    public void calculate(String activationFunction) {
        for(Neuron n:neurons){
        	if(!n.getDropout()){
            n.calculate(previousLayer, activationFunction);
        } else {
        	n.setOutputValue(0);
        	n.setOutputDerivative(0);
        	}
        }
    }

    public void updateWeights(double eta) {
        for(Neuron n:neurons){
            n.updateWeights(previousLayer, eta);
        }
    }

    public void backwardspropagation() {
        for(Neuron n:neurons){
            n.backprop(nextLayer);
        }
    }

    public void calculateOutputErrorDerivative(double... exp) {
        for(int i = 0; i < neurons_amount; i++) {
            neurons[i].setErrorValue(-(exp[i] - neurons[i].getOutputValue()) * neurons[i].getOutputDerivative());
        }
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
}
