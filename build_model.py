from keras.models import Sequential
from image_data_generator import DataGenerator
from keras.applications import InceptionV3
from prepare_data import PrepareData
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from keras import layers 
import keras
import matplotlib.pyplot as plt
import numpy as np


class BuildModel:
    def __init__(self, new=False):
        self.model = None
        if new:
            base_model = InceptionV3(
                include_top=False,
                weights='imagenet',
                input_shape=(178, 178, 3))

            model = Sequential()
            model.add(base_model)
            model.add(layers.GlobalAveragePooling2D())
            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu'))
            model.add(layers.Dense(1, activation='sigmoid'))

            for layer in base_model.layers:
                # if isinstance(layer, layers.BatchNormalization):
                #     layer.trainable = True
                layer.trainable = False

            model.compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy']
            )

            self.model = model
        

    @staticmethod
    def _transform_data(data):
        return data / 255

    def _transform_labels():
        pass

    def train(self, x_train, y_train, x_valid, y_valid, epochs=10, augment=False):
        assert self.load_model is not None, 'create new model before training'

        x_valid = self._transform_data(x_valid)

        train_gen = DataGenerator(
            images=x_train, 
            labels=y_train, 
            batch_size=64, 
            shuffle=False, 
            augment=augment
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            verbose=True,
            patience=2
        )

        self.model.fit(
            train_gen, 
            epochs=epochs, 
            verbose=True,
            shuffle=True, 
            validation_data=(x_valid, y_valid),
            callbacks=[early_stop]
        )

    def single_prediction(self, data):
        """Returns probability"""
        data = data / 255
        prediction = self.model.predict(np.array([data, ]), verbose=False)
        return prediction

    def save_model(self, name, path='./saved_models/'):
        self.model.save(f'{path+name}')

    def load_model(self, name, path='./saved_models/'):
        self.model = keras.models.load_model(f'{path+name}')

    @staticmethod
    def _plot_history(history):
        #Plot the Loss Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'],'r',linewidth=3.0)
        plt.plot(history.history['val_loss'],'b',linewidth=3.0)
        plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Loss',fontsize=16)
        plt.title('Loss Curves',fontsize=16)
        #Plot the Accuracy Curves
        plt.figure(figsize=[8,6])
        plt.plot(history.history['accuracy'], 'r', linewidth=3.0)
        plt.plot(history.history['val_accuracy'], 'b',linewidth=3.0)
        plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
        plt.xlabel('Epochs ',fontsize=16)
        plt.ylabel('Accuracy',fontsize=16)
        plt.title('Accuracy Curves',fontsize=16)

    def plot_history(self):
        self._plot_history(self.model.history)

    def plot_auc(self, x_test, y_test, treshold=0.5):
        x_test = x_test / 255
        y_pred = self.model.predict(x_test)
        y_pred_treshold = np.where(y_pred > treshold, 1, 0)

        
        print(classification_report(y_test, y_pred_treshold))
        RocCurveDisplay.from_predictions(y_test, y_pred_treshold)

    

if __name__ == '__main__':
    pass
    # my_model = BuildModel(True)
    
    # x_train, y_train = PrepareData.load_set('train', './data/processed/')
    # x_valid, y_valid = PrepareData.load_set('valid', './data/processed/')

    # my_model.train(x_train, y_train, x_valid, y_valid, augment=False)
    # my_model.save_model('binary_model1')
