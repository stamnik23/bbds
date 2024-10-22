import os
import mne
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import EarlyStopping

base_dir = r'C:\Users\nstamo\bb\ds005555-download'

def load_subject_data(subject_folder, sfreq=100, epoch_size=7680):
    eeg_dir = os.path.join(base_dir, subject_folder, 'eeg')
    edf_file = os.path.join(eeg_dir, f'{subject_folder}_task-Sleep_acq-headband_eeg.edf')
    tsv_file = os.path.join(eeg_dir, f'{subject_folder}_task-Sleep_acq-headband_events.tsv')
    
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    data, times = raw.get_data(return_times=True)
    avg_data = np.mean(data, axis=0)
    labels = pd.read_csv(tsv_file, sep='\t')['ai_hb'].values
    
    num_epochs = len(avg_data) // epoch_size
    if len(labels) != num_epochs:
        print(f"Mismatch: {len(labels)} labels and {num_epochs} epochs in {subject_folder}.")
        return None, None
    
    epochs = avg_data[:num_epochs * epoch_size].reshape((num_epochs, epoch_size))
    valid_indices = labels != -2
    epochs = epochs[valid_indices]
    labels = labels[valid_indices]
    
    return epochs[..., np.newaxis], labels

def load_all_subjects(base_dir):
    all_data = []
    all_labels = []
    
    for subject_folder in os.listdir(base_dir):
        subject_dir = os.path.join(base_dir, subject_folder)
        if os.path.isdir(subject_dir) and subject_folder.startswith('sub-') and subject_folder != 'sub-99':
            print(f"Processing {subject_folder}...")
            data, labels = load_subject_data(subject_folder)
            if data is not None and labels is not None:
                all_data.append(data)
                all_labels.append(labels)
    
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print(f"Total data shape: {all_data.shape}, Total labels shape: {all_labels.shape}")
    return all_data, all_labels

def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape))
    model.add(layers.Conv1D(16, kernel_size=5, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.SpatialDropout1D(rate=0.01))

    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.SpatialDropout1D(rate=0.01))

    model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
    model.add(layers.Conv1D(256, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.SpatialDropout1D(rate=0.01))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

data, labels = load_all_subjects(base_dir)
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1], 1)
model = create_cnn_model(input_shape)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
score = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {score[1]}")
model.save('cnn_sleep_model.h5')
