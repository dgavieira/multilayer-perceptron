from preprocess.data_processor import DataProcessor
from train.cross_validation import CrossValidator
import tensorflow as tf
from models.model_1 import MyModel
from models.model_2 import MyModel2
from visualization.plotter import ModelPlotter
from evaluator.evaluator import ModelEvaluator



def main():
    # Fetch and preprocess data
    data_processor = DataProcessor(dataset_id=161)
    data_processor.select_features(['Age', 'Shape', 'Margin', 'Density'], 'Severity')
    data_processor.handle_missing_values()
    data_processor.standardize_data()
    X, y = data_processor.X, data_processor.y

     # Instantiate models
    model_1 = MyModel(input_shape=X.shape[1])
    model_1.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    model_2 = MyModel2(input_shape=X.shape[1])
    model_2.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Cross-validation and evaluation for model 1
    cross_validator = CrossValidator(model_1)
    cross_validator.cross_validate(X, y)
    print('Arquitetura 1 - Acurácia Média:', cross_validator.mean_accuracy())

    # Cross-validation and evaluation for model 2
    cross_validator = CrossValidator(model_2)
    cross_validator.cross_validate(X, y)
    print('Arquitetura 2 - Acurácia Média:', cross_validator.mean_accuracy())

    # Detailed evaluation for model 1
    model_evaluator = ModelEvaluator(cross_validator.kfold)
    model_evaluator.evaluate_model(model_1, X, y)

    # Detailed evaluation for model 2
    model_evaluator.evaluate_model(model_2, X, y)

    # Plot convergence for model 1
    ModelPlotter.plot_convergence(model_1, X, y)

    # Plot convergence for model 2
    ModelPlotter.plot_convergence(model_2, X, y)

    # Plot model architectures
    ModelPlotter.plot_model_architecture(model_1, 'model_1.png')
    ModelPlotter.plot_model_architecture(model_2, 'model_2.png')

if __name__ == "__main__":
    main()
