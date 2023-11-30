# import tensorflow as tf

# def test_tensorflow_gpu():
#     # Check if TensorFlow can access GPU
#     gpus = tf.config.list_physical_devices('GPU')
    
#     if gpus:
#         print("GPU is available.")
        
#         # Print information about available GPUs
#         for gpu in gpus:
#             print(f"Device name: {gpu.name}")
#             print(f"Device type: {gpu.device_type}")
#     else:
#         print("GPU is not available. TensorFlow will use CPU.")

# if __name__ == "__main__":
#     test_tensorflow_gpu()


import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the name of the current GPU
    current_gpu = torch.cuda.get_device_name(0)
    print(f"Current GPU: {current_gpu}")
else:
    print("CUDA is not available. Running on CPU.")
