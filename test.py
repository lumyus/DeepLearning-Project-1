from models import advanced_cnn as advanced_cnn
from models import simple_cnn as simple_cnn
from models import siamese_cnn as siamese_cnn
from models import siamese_cnn_weightsharing as siamese_cnn_weightsharing
from utils import utils as utils

# Parameters for all models
print_epochs = 5
image_pairs = 1000

batch_size = 100
hidden_layers = 128
epochs = 25

# Description
# The SimpleCNN contains 2 inputs and 1 output. It compares the two images in the two channels and makes a prediction
# straight away to determine for each pair if the first digit is lesser or equal to the second

# The AdvancedCNN  contains 1 input and 10 outputs. It separates the two channels and predicts separately the value of
# each image. Logic comparison is subsequently used to determine for each pair if the first digit is lesser or equal
# to the second

if __name__ == "__main__":
    print('\nTraining and Testing all models.')

    # The handling function takes care of the training as well as of the testing of the models

    print('\nThe SimpleCNN is being trained and tested...')
    testing_accuracy_simple_cnn, training_accuracy_simple_cnn, validation_accuracy_simple_cnn = utils.handle_simple_cnn(
        image_pairs, batch_size, epochs, print_epochs, hidden_layers,
        simple_cnn.SimpleConvolutionalNeuralNetwork)

    print('\nThe AdvancedCNN is being trained and tested...')
    testing_accuracy_advanced_cnn, training_accuracy_advanced_cnn, validation_accuracy_advanced_cnn = utils.handle_advanced_cnn(
        image_pairs, batch_size, epochs, print_epochs, hidden_layers,
        advanced_cnn.AdvancedConvolutionalNeuralNetwork)
    
    print('\nThe SiameseCNN is being trained and tested...')
    testing_accuracy_siamese_cnn, training_accuracy_siamese_cnn, validation_accuracy_siamese_cnn = utils.handle_siamese(
        image_pairs, batch_size, epochs, print_epochs, hidden_layers,
        siamese_cnn.SiameseConvolutionalNeuralNetwork)
    
    print('\nThe SiameseWeightsharingCNN is being trained and tested...')
    testing_accuracy_siamese_weightsharing_cnn, training_accuracy_siamese_weightsharing_cnn, validation_accuracy_siamese_weightsharing_cnn = utils.handle_siamese(
        image_pairs, batch_size, epochs, print_epochs, hidden_layers,
        siamese_cnn_weightsharing.SiameseWeightsharingConvolutionalNeuralNetwork)
    

    print('\nTraining and Testing for all models has been completed!')
    print('Testing 1000 pairs resulted in the following accuracies:')
    print(f'SimpleCNN : {testing_accuracy_simple_cnn:.2f}%')
    print(f'AdvancedCNN : {testing_accuracy_advanced_cnn:.2f}%')
    print(f'SiameseCNN : {testing_accuracy_siamese_cnn:.2f}%')
    print(f'SiameseWeightsharingCNN : {testing_accuracy_siamese_weightsharing_cnn:.2f}%')


