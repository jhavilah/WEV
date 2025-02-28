import tensorflow as tf
import numpy as np
import os

def create_model():
    """Cria e compila o modelo R1 para classificação."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,), name='input'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def representative_data_gen():
    """Gera dados fictícios para calibração de quantização (substitua por dados reais)."""
    for _ in range(100):
        yield [np.random.uniform(0, 1, (1, 32)).astype(np.float32)]

def convert_to_tflite(model, quantize=False):
    """Converte o modelo para formato TFLite com/sem quantização."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Configuração para quantização INT8
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        converter.quantized_input_stats = {0: (0.0, 1.0)}
        model_name = "R1_model_int8.tflite"
    else:
        model_name = "R1_model_float32.tflite"
    
    tflite_model = converter.convert()
    
    with open(model_name, "wb") as f:
        f.write(tflite_model)
    
    print(f"Modelo salvo: {model_name} ({os.path.getsize(model_name)/1024:.2f} KB)")
    return model_name

def test_inference(model_path):
    """Testa a inferência do modelo TFLite."""
    try:
        from tflite_runtime.interpreter import Interpreter  # Para edge devices
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter  # Fallback
    
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Obter detalhes de entrada/saída
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preparar dados de entrada (substitua por dados reais)
    input_shape = input_details[0]['shape']
    input_data = np.random.uniform(0, 1, input_shape).astype(
        np.uint8 if 'int8' in model_path else np.float32
    )
    
    # Executar inferência
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Obter resultados
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"\nPredição do modelo ({os.path.basename(model_path)}):")
    print(f"Classe prevista: {np.argmax(output[0])}")
    print(f"Probabilidades: {output[0]}")

if __name__ == "__main__":
    # Treinar e converter modelos
    model = create_model()
    print("\nConvertendo modelo float32...")
    float_model = convert_to_tflite(model, quantize=False)
    
    print("\nConvertendo modelo int8 quantizado...")
    quant_model = convert_to_tflite(model, quantize=True)
    
    # Testar inferência
    print("\nTestando float32:")
    test_inference(float_model)
    
    print("\nTestando int8:")
    test_inference(quant_model)
