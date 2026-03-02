# zero-model
This code is train a model from zero  
# zero-model  
  
This code trains a model from zero.  
  
This is a Word2Vec (CBOW concept) word vector training and prediction system written in NumPy. It includes a training script and an interactive prediction script.  
  
📂 Project Structure  
  
Code 1 (Training Script): Reads the corpus, updates word vectors and weights through backpropagation, and saves the model as .json and .npz files.  
  
Code 2 (Inference Script): Loads the trained model parameters and predicts the next possible character based on the input text.  
  
🛠️ Environment Dependencies  
  
Python 3.x  
  
NumPy  
  
📝 Data Preparation   
  
Before running, ensure the following files are in the directory:  
  
output.txt: The original text corpus used for training.  
  
Tokenizer-full.json: The vocabulary file (formatted as {"character": ID}).  
  
🚀 User Guide  
  
1. Training the Model  
  
Run the training script to generate word vectors: `python train.py`  
  
Training Principle: The script reads each line of text, using all words preceding the current character as context to predict the next word.  
  
Output Files:  
  
`model_weights.npz`: Stores the output layer weights `W_out` and biases `b_out`.  
  
`word2vec.json`: Stores the trained word vectors.  
  
2. Text Prediction    
  
After the model training is complete, run the inference script for interaction: `python predict.py`  
  
Function: Input a sentence, and the program will calculate the top-5 candidate characters with the highest probability based on the average word vectors.  
  
⚙️ Hyperparameter Tuning  
  
You can modify the following parameters in the code to optimize the effect:  
  
`D = 128`: The dimension of the word vectors.  
  
`lr = 0.05`: The learning rate.  
  
`epoch_times = 10`: The number of training epochs.  
