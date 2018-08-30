import java.util.Arrays;

import CrossNetworkFeatures.DataHandler;
import NeuralNetwork.NeuralNetwork;
import ObjectOrientatedNN.Network;

public class Example {

	public static void main(String args[]){
	    	Network network = new Network("Network "  ,2, 10, 4 , 1);
	    	DataHandler dataHandler = new DataHandler();
	    	dataHandler.insertData(new double[]{1, 2}, new double[] {0.03} );
	    	dataHandler.insertData(new double[]{4, 1}, new double[] {0.05} );
	    	dataHandler.insertData(new double[]{0, 1}, new double[] {0.01} );
	    	dataHandler.insertData(new double[]{1, 1}, new double[] {0.02} );
	    	dataHandler.insertData(new double[]{5, 3}, new double[] {0.08} );
	    	dataHandler.insertData(new double[]{5, 2}, new double[] {0.07} );
	    	dataHandler.insertData(new double[]{1, 6}, new double[] {0.07} );
            dataHandler.insertData(new double[]{1, 1}, new double[]{0.02});
            dataHandler.insertData(new double[]{2, 2}, new double[]{0.04});
            dataHandler.insertData(new double[]{1, 2}, new double[]{0.03});
            dataHandler.insertData(new double[]{3, 4}, new double[]{0.07});
            
	    	network.setActivationLeakyRelu();
	    	network.setDropOut(true);
	    	network.setDropoutRatio(0.4);
	    	network.setModulusNumber(3);
	    	network.setModulusNumber(1239003);

	    	//Before Training values
	        System.out.println(Arrays.toString(network.calculate(new double[]{2, 6})));
	        System.out.println(Arrays.toString(network.calculate(new double[]{4, 1})));
        	network.train(dataHandler, 100000, 7, 0.3);

	        for(int i = 0; i < 4; i++) {
	        	System.out.println(Arrays.toString(dataHandler.getInput(i))+ " Calculated " + Arrays.toString(network.calculate(dataHandler.getInput(i))));
	        }
	        
	        //After Training values
	        System.out.println(Arrays.toString(network.calculate(new double[]{2, 6})));
	        System.out.println(Arrays.toString(network.calculate(new double[]{4, 1})));
	}
}
	        


	        
