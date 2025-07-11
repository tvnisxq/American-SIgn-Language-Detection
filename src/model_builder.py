from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation = 'relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activator = 'relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation = 'relu'),
        Dropout(0.5),
        Dense(num_classes, activation = 'softmax')

    ])
    return model