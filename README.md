MachineLearning

****************************************************************************************************************
                                            Designing the DNN guidelines.
****************************************************************************************************************
-Rule 1: Start with small architectures.

    In the case of DNNs, it is always advised to start with a single-layer network with around 100–300 neurons.
    Train the network and measure performance using the defined metrics (while defining the baseline score). 
    If the results are not encouraging, try adding one more layer with the same number of neurons and repeating the process.

- Rule 2: When small architectures (with two layers) fail, increase the size.

    When the results from small networks are not great, you need to increase the number of neurons
    in layers by three to five times (i.e., around 1,000 neurons in each layer). Also, increase regularization
    to 0.3, 0.4, or 0.5 for both layers and repeat the process for training and performance measurement. 

-Rule 3: When larger networks with two layers fail, go deeper.
    
    Try increasing the depth of the network with more and more layers while maintaining regularization with
    dropout layers (if required) after each dense (or your selected layer) with a dropout rate between 0.2 and 0.5.

-Rule 4: When larger and deeper networks also fail, go even larger and even deeper.

    In case large networks with ~1000 neurons and five or six layers also don’t give the desired performance,try increasing 
    the width and depth of the network. Try adding layers with 8,000–10,000 neurons per layer and a dropout of 0.6 to 0.8.
	
-Rule 5: When everything fails, revisit the data.

    If all the aforementioned rules fail, revisit the data for improved feature engineering and
    normalization, and then you will need to try other ML alternatives.
*****************************************************************************************************************
