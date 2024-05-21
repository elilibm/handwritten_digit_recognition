# handwritten_digit_recognition
Trained a neural network to classify handwritten digits using PyTorch.

# here is how it works ...

Step 1: Importing the dependencies.
Import torch and torchvision. 

Step 2: Calling and transforming the dataset used for training. 
Calls the MNIST dataset (handwritten digits used for training image processing systems) which is part of the torchvision library. 

Determines where the data will be stored, whether it will load the train or test data with a bool, converts images into tensors (multi-dimensional arrays) which is suitable for neutral nets, then downloads the data from the internet. 

Step 3: Creating loaders for faster, more efficient dataset load time. 
Import DataLoader which is a utility in PyTorch to efficiently load data in batches. 

Creates two loaders that load the MNIST dataset, specifies that each batch will have 100 images, shuffles the data at each step so the model does not learn the order of the data (affects accuracy), then assigns one additional subprocess which can help load the data. 

Step 4: Creating the architecture for the neural net. 
Import the required libraries. 

Defines a CNN class that inherits from nn.Module which is a base class for all neural network modules in PyTorch. 

Defines a constructor method _init_ which ensures that the base class (nn.Module),  inherited from CNN, is properly initialized. 

Creates conv1, a convolutional layer that takes the image and then uses 10 different 5x5 kernels to produce 10 feature maps. The output will be 10 maps/channels, each with different features detected by the kernels. These 10 output channels are then used to create 20 output channels in the second convolutional layer conv2. 

Creates conv2_drop, a dropout layer that randomly zeros out entire channels to prevent model overfitting (learns the model too well including noise and outliers which is bad when using new data). 

Creates two linear layers that use nn.Linear to create linear transformations that manipulate layers of input into fewer layers of output. First takes 320 input and turns into 50 output then the second layer takes that 50 as input and turns it into 10 output for digits (0-9).

Creates the forward method which defines how the input tensor (way in which input is stored) flows through the neural net. This method is provided by the CNN class. 

Creates step that applies 2D max pooling (finding the maximum value in a sub-array and using that in a new one) with a 2 x 2 window to the output from conv1, reducing the dimensions of the feature maps by 2 for more efficiency, while keeping the most significant features. Then passes this through relu (the Rectified Linear Unit activation function) which adds non-linearity into the network so it can learn more complex patterns and removes negative values so fewer neurons are activated and the network is more efficient. Then applies the output from this layer as the input that goes through conv2 and max_pool and F.relu to produce a smaller, more precise output. 

Flatten this output (turn 2D to 1D) by using .view to reshape the tensor. The -1 and 320 reshape the tensor x to have 320 features per image, with the first dimension inferred automatically based on the batch size (-1 is a placeholder for that size).

Applies flattened output as input to fc1 transforming the 320 input into 50 output. Since it uses a linear transformation, use relu to reapply non-linearity. Dropout is applied to this output to prevent overfitting. That new output is sent into fc2 to transform the data into 10 output features. 

Returns F.softmax of previous output, an activation function which converts raw output scores into probabilities corresponding to specific classes (representing specific digits). Outputs probabilities of the likelihood of image corresponding to specific digits. 

Step 5: Computation set up. 
Checks if GPU (cuda) is available to process and train the neural net since it is faster and if it is, applies training to that. If not, does training on CPU. 

Step 6:
