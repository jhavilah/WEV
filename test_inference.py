import numpy as np

def test_inference(model_path):
    """Testa a inferência do modelo TFLite."""
    try:
        # Tenta usar runtime otimizado para edge
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # Fallback para TensorFlow completo
        from tensorflow.lite.python.interpreter import Interpreter
    
    # Carregar modelo
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Obter detalhes de entrada/saída
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Gerar dados de teste (substitua por dados reais)
    input_shape = input_details[0]['shape']
    input_data = np.random.uniform(0, 1, input_shape).astype(
        np.uint8 if 'int8' in model_path else np.float32
    )
    
    # Executar inferência
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Obter resultados
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"\nResultado para {model_path}:")
    print(f"Classe prevista: {np.argmax(output)}")
    print(f"Confiança: {np.max(output):.2%}")

if __name__ == "__main__":
    # Testar ambos os modelos
    test_inference("R1_model_float32.tflite")
    test_inference("R1_model_int8.tflite")
