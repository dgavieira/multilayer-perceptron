import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

class ModelPlotter:
    @staticmethod
    def plot_convergence(model_creator, X, y):
        model = model_creator()
        history = model.fit(X, y, epochs=50, batch_size=10, verbose=0, validation_split=0.2)
        plt.plot(history.history['accuracy'], label='Treinamento')
        plt.plot(history.history['val_accuracy'], label='Validação')
        plt.title('Curva de Convergência da Acurácia')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_model_architecture(model_creator, filename):
        model = model_creator()
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=False)

# Example usage
def create_model_1():
    model = Sequential([
        Dense(10, activation='relu', input_shape=(X.shape[1],)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_model_2():
    model = Sequential([
        Dense(20, activation='relu', input_shape=(X.shape[1],)),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Assuming X and y are already defined and are numpy arrays
plotter_1 = ConvergencePlotter(create_model_1)
plotter_1.plot_convergence(X, y)

plotter_2 = ConvergencePlotter(create_model_2)
plotter_2.plot_convergence(X, y)
