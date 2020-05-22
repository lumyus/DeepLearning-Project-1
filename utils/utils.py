import torch
from torch import nn
from torch import optim
import time
from utils import dlc_practical_prologue as prologue


def train_simple_cnn(model, train_input, train_target, validation_input, validation_target, device, epochs,
                     batch_size, print_epochs):
    print("\nStarting to train the SimpleCNN!")

    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()

    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Capture historical losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []
    loss = 0

    for epoch in range(epochs + 1):

        for batch in range(0, train_input.size(0), batch_size):  # train_input.size(0) = 900

            output = model(train_input.narrow(0, batch, batch_size))

            loss = criterion(output, train_target.narrow(0, batch, batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create accuracy statistics at a print_step frequency
        if epoch % print_epochs == 0:
            training_loss.append(loss.item())
            print(f'\nEpoch : {epoch}, Loss: {loss.item():.4f}')

            # Change mode to testing
            model.eval()

            # Calculate training accuracy and print

            accuracy = 100 - 100 * (
                    calculate_incorrect_classifications_simple_cnn(model, train_input, train_target, batch_size)
                    / (train_input.size(0)))

            training_accuracy.append(accuracy)

            print(f'Training Accuracy : {accuracy:.4f}%')

            # Calculate validation accuracy and print

            accuracy = 100 - 100 * (
                    calculate_incorrect_classifications_simple_cnn(model, validation_input, validation_target,
                                                                   batch_size) / (
                        validation_input.size(0)))
            validation_accuracy.append(accuracy)

            print(f'Validation Accuracy : {accuracy:.4f}%')

            # Change mode back to training
            model.train()

    return model, training_loss, training_accuracy, validation_accuracy


def train_model_advanced_cnn(model, train_input, train_target, train_classes, validation_input, validation_target,
                             device, epochs, batch_size, print_epochs):
    print("\nStarting to train the AdvancedCNN!")

    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()

    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Capture historical losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []
    loss = 0

    for epoch in range(epochs + 1):

        for i in range(0, train_input.size(0), batch_size):
            output = model(train_input.narrow(0, i, batch_size))

            loss = criterion(output, train_classes.narrow(0, i, batch_size))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create accuracy statistics at a print_step frequency
        if epoch % print_epochs == 0:
            training_loss.append(loss.item())
            print(f'\nEpoch : {epoch}, Loss: {loss.item():.4f}')

            # Change mode to testing
            model.eval()

            # Calculate training accuracy and print

            accuracy = 100 - 100 * (calculate_incorrect_classifications_advanced_cnn(model, train_input, train_target,
                                                                                     batch_size) / (
                                        train_input.size(0)))
            training_accuracy.append(accuracy)

            print(f'Training Accuracy : {accuracy:.4f}%')

            # Calculate validation accuracy and print

            accuracy = 100 - 100 * (
                    calculate_incorrect_classifications_advanced_cnn(model, validation_input, validation_target,
                                                                     batch_size) / (
                        validation_input.size(0)))
            validation_accuracy.append(accuracy)

            print(f'Validation Accuracy : {accuracy:.4f}%')

            # Change mode back to training
            model.train()

    return model, training_loss, training_accuracy, validation_accuracy


def handle_simple_cnn(image_pairs, batch_size, epochs, print_epochs, hidden_layers, simple_cnn):
    if torch.cuda.is_available():
        print('\nUsing GPU...\n')
        device = torch.device('cuda')
    else:
        print('\nUsing CPU...\n')
        device = torch.device('cpu')

    model = simple_cnn(hidden_layers)

    # Print model characteristics
    print(model)

    # Load model onto device
    model.to(device)

    # Generate training and testing dataset
    train_input, train_target, train_classes, test_input, test_target, test_classes \
        = prologue.generate_pair_sets(image_pairs)

    # Calculate data mean and standard deviation
    mean = train_input.mean()
    std = train_input.std()

    # Normalize the data (following dlc_practical_proloque.py example)
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Following https://stackoverflow.com/
    # /questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    # Split the training and testing into is not done. We keep 1000 testing pairs since there is enough data available
    # Split the training data into training and validation at 80%:20%
    # Example: 1000 pairs would result in 800 training pairs and 200 validation pairs

    validation_input = train_input[800:1000]
    validation_target = train_target[800:1000]

    train_input = train_input[0:800]
    train_target = train_target[0:800]
    
    validation_classes = train_classes[800:1000]
    
    # Use eventually cuda
    train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
    validation_input, validation_target, validation_classes = validation_input.to(device), validation_target.to(device), validation_classes.to(device)
    test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)
    
    # Train model
    model, training_loss, training_accuracy, validation_accuracy \
        = train_simple_cnn(model, train_input, train_target, validation_input, validation_target,
                           device, epochs, batch_size, print_epochs)

    print(f'\nTraining completed!')
    print(f'\nTesting started...')

    # Change mode to testing again. This time to do a final performance test.
    # The testing will be done on a unseen dataset
    # Previously model.eval() was called to test during model training

    model.eval()

    errors_in_testing = calculate_incorrect_classifications_simple_cnn(model, test_input, test_target, batch_size)

    total_testing = test_input.size(0)
    error_rate = 100 * errors_in_testing / total_testing

    testing_accuracy = 100 - error_rate

    print(f'\nTesting Completed!')

    return testing_accuracy, training_accuracy, validation_accuracy


def handle_advanced_cnn(image_pairs, batch_size, epochs, print_epochs, hidden_layers, advanced_cnn):
    if torch.cuda.is_available():
        print("Using GPU!")
        device = torch.device('cuda')
    else:
        print("Using CPU!")
        device = torch.device('cpu')

    model = advanced_cnn(hidden_layers)

    # Print model characteristics
    print(model)

    # Load model onto device
    model.to(device)

    # Generate training and testing dataset
    train_input, train_target, train_classes, test_input, test_target, test_classes \
        = prologue.generate_pair_sets(image_pairs)

    # Calculate data mean and standard deviation
    mean = train_input.mean()
    std = train_input.std()

    # Normalize the data (following dlc_practical_proloque.py example)
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Following https://stackoverflow.com/
    # /questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    # Split the training and testing into is not done. We keep 1000 testing pairs since there is enough data available
    # Split the training data into training and validation at 80%:20%
    # The two channels of each pair are split into individual two digit classes

    train_input = train_input.view(-1, 1, 14, 14)
    test_input = test_input.view(-1, 1, 14, 14)

    validation_input = train_input[1600:2000]
    validation_target = train_target[800:1000]

    train_input = train_input[0:1600]
    train_target = train_target[0:800]
    

    train_classes = train_classes.view(-1, 1)
    train_classes = train_classes[0:1600]
    train_classes = train_classes.reshape((-1,))
    
    # Use eventually cuda
    train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
    validation_input, validation_target = validation_input.to(device), validation_target.to(device)
    test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)
    
    # Train model
    model, training_loss, training_accuracy, validation_accuracy \
        = train_model_advanced_cnn(model, train_input, train_target, train_classes, validation_input, validation_target,
                                   device, epochs, batch_size, print_epochs)

    print(f'\nTraining completed!')
    print(f'\nTesting started...')
    # Change mode to testing again. This time to do a final performance test.
    # The testing will be done on a unseen dataset
    # Previously model.eval() was called to test during model training

    model.eval()

    errors_in_testing = calculate_incorrect_classifications_advanced_cnn(model, test_input, test_target, batch_size)
    total_testing = test_input.size(0)
    error_rate = 100 * errors_in_testing / (
            total_testing / 2)  # The error has to be divided by 2 because of the split of the two input channels

    testing_accuracy = 100 - error_rate

    print(f'\nTesting Completed!')

    return testing_accuracy, training_accuracy, validation_accuracy


def calculate_incorrect_classifications_simple_cnn(model, input, target, batch_size):
    incorrect_classifications = 0  # Amount of incorrect classifications

    for batch in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, batch, batch_size))
        _, predicted_classes = torch.max(output.data, 1)

        for i in range(batch_size):
            if target.data[batch + i] != predicted_classes[i]:
                incorrect_classifications += 1

    return incorrect_classifications


def calculate_incorrect_classifications_advanced_cnn(model, input, target, batch_size):
    incorrect_classifications = 0  # Amount of incorrect classifications
    first_channel = []  # First channel testing
    second_channel = []  # Second channel testing
    predictions = []

    # Get the prediction for each channel
    for batch in range(0, input.size(0), batch_size):
        output = model(input.narrow(0, batch, batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for i in range(batch_size):
            if i % 2 == 0:  # Load the testing results into two arrays corresponding to the two channels
                first_channel.append(predicted_classes[i])
            else:
                second_channel.append(predicted_classes[i])

    # Compare if the first channel's number is greater than the second channel's (purely logical)
    for i in range(len(target)):

        if first_channel[i] > second_channel[i]:
            predictions.append(0)
        else:
            predictions.append(1)

        # Evaluate the result with the target result
        if target.data[i] != predictions[i]:
            incorrect_classifications += 1

    return incorrect_classifications
def train_siamese(model, train_input, train_target, train_classes, validation_input, validation_target, 
                  device, epochs, batch_size, print_epochs):
    print("\nStarting to train the Siamese CNN!")
    
    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()

    

    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Capture historical losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []
    time_start = time.time()
    

    for epoch in range(epochs+1):
        
            
        for batch in range(0, train_input.size(0), batch_size):
            
            out_a, out_b = model(train_input.narrow(0, batch, batch_size))
            
            out_a_loss = criterion(out_a, train_classes[:, 0].narrow(0, batch, batch_size)) # Alternative 1
            
            out_b_loss = criterion(out_b, train_classes[:, 1].narrow(0, batch, batch_size)) # Alternative 1
            
            loss = 0.5*out_a_loss + 0.5*out_b_loss # Alternative 1
          
            model.zero_grad()
            loss.backward()
            optimizer.step()

        # Create accuracy statistics at a print_epochs frequency            
        if epoch % print_epochs == 0:
            training_loss.append(loss.item())
            print(f'\nEpoc : {epoch}, Loss a : {out_a_loss.item()} Loss b : {out_b_loss.item()} Loss: {loss.item()}')
            print(f'\nTime lapsed: {(time.time() - time_start)}')

            # Change mode to testing
            model.eval()


             # Calculate training accuracy and print

            accuracy = 100 - 100 * (calculate_incorrect_classifications_siamese_cnn(model, train_input, train_target,
                                                                                     batch_size) / (
                                        train_input.size(0)))
            training_accuracy.append(accuracy)

            print(f'Training Accuracy : {accuracy:.4f}%')

            # Calculate validation accuracy and print

            accuracy = 100 - 100 * (
                    calculate_incorrect_classifications_siamese_cnn(model, validation_input, validation_target,
                                                                     batch_size) / (
                        validation_input.size(0)))
            validation_accuracy.append(accuracy)

            print(f'Validation Accuracy : {accuracy:.4f}%')


            # Change mode back to training
            model.train()
    return model, training_loss, training_accuracy, validation_accuracy




def handle_siamese(image_pairs, batch_size, epochs, print_epochs, hidden_layers, siamese_cnn):
    if torch.cuda.is_available():
        print('\nUsing GPU...\n')
        device = torch.device('cuda')
    else:
        print('\nUsing CPU...\n')
        device = torch.device('cpu')
        
    model = siamese_cnn(hidden_layers)
    
    # Print model characteristics
    print(model)
    
    # Load model onto device
    model.to(device)

    # Generate training and testing dataset
    train_input, train_target, train_classes, test_input, test_target, test_classes \
    = prologue.generate_pair_sets(image_pairs)
    
    # Calculate data mean and standard deviation    
    mean = train_input.mean()  
    std = train_input.std()
    
    
    # Normalize the data (following dlc_practical_proloque.py example)
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Following https://stackoverflow.com/
    # /questions/13610074/is-there-a-rule-of-thumb-for-how-to-divide-a-dataset-into-training-and-validatio
    # Split the training and testing into is not done. We keep 1000 testing pairs since there is enough data available
    # Split the training data into training and validation at 80%:20%
    # Example: 1000 pairs would result in 800 training pairs and 200 validation pairs

    validation_input = train_input[800:1000]
    validation_target = train_target[800:1000]

    train_input = train_input[0:800]
    train_target = train_target[0:800]
    
    train_classes = train_classes[0:800]

    
    # Use eventually cuda
    train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
    validation_input, validation_target = validation_input.to(device), validation_target.to(device)
    test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)
    
    # Train model
    model, training_loss, training_accuracy, validation_accuracy \
        = train_siamese(model, train_input, train_target, train_classes, validation_input, validation_target,
                         device, epochs, batch_size, print_epochs)

    print(f'\nTraining completed!')
    print(f'\nTesting started...')

    # Change mode to testing again. This time to do a final performance test.
    # The testing will be done on a unseen dataset
    # Previously model.eval() was called to test during model training

    model.eval()
 
    errors_in_testing = calculate_incorrect_classifications_siamese_cnn(model, test_input, test_target, batch_size)
    total_testing = test_input.size(0)
    error_rate = 100 * errors_in_testing / (
            total_testing / 2)  # The error has to be divided by 2 because of the split of the two input channels

    testing_accuracy = 100 - error_rate

    print(f'\nTesting Completed!')

    return testing_accuracy, training_accuracy, validation_accuracy

def calculate_incorrect_classifications_siamese_cnn(model, data_input, data_target, batch_size):

    
    incorrect_classifications = 0 # Amount of incorrect classifications
    
    prediction_a = []
    prediction_b = []   
    prediction = []
    
    for batch in range(0, data_input.size(0), batch_size):

        out_a, out_b = model(data_input.narrow(0, batch, batch_size))
        _, predicted_classes_a = torch.max(out_a.data, 1)
        _, predicted_classes_b = torch.max(out_b.data, 1)
        
        for k in range(batch_size): # 10 classes prediction (number value)
            prediction_a.append(predicted_classes_a[k]) 
            prediction_b.append(predicted_classes_b[k])
            
    for x in range(len(data_target)):
        
        #  Aim to predict whether first digit is lesser or equal to the second (Guideline)
        if(prediction_a[x] > prediction_b[x]):
            prediction.append(0) # superior
        else:
            prediction.append(1) # less or equal

        if data_target.data[x] != prediction[x]:
            incorrect_classifications += 1 # Error counter

    return incorrect_classifications
