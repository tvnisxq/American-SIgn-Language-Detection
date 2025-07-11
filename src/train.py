from src.data_loader import get_data_generators
from src.model_builder import create_cnn_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

# Load data 
train_gen, val_gen = get_data_generators('data/raw/asl_alphabet_train')

# Model config
input_shape = (64, 64, 3)
num_classes = train_gen.num_classes
model = create_cnn_model(input_shape, num_classes)

# Compile
model.compile(optimizer=Adam(),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

# Save best model
checkpoint = model.checkpoint('artifacts/models/asl_cnn.h5',save_best_only = True)

# Train
model.fit(
    train_gen,
    validation_data = val_gen,
    epochs = 10,
    callbacks = [checkpoint]
)