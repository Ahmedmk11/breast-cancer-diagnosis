'''
This script is used to create a GUI application for multi-stage diagnosis of breast cancer.
The application allows the user to upload an image and then displays the diagnosis result.

'''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import threading
import os

def predict(image_path: str, shape: tuple, model_path: str) -> int:
    '''
    This function is used to predict the class of an image using a tflite model.

    :param image_path: The path to the image
    :param shape: The shape of the image
    :param model_path: The path to the tflite model
    :return: The predicted class
    '''

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = load_img(image_path, target_size=shape)
    image = img_to_array(image) / 255.0
    image = image[np.newaxis, ...]
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = 1 if predictions[0][0] > 0.5 else 0
    return predicted_class

def diagnose(image_path: str) -> str:
    '''
    This function is used to diagnose an image using two tflite models.

    :param image_path: The path to the image
    :return: The diagnosis result
    '''

    shape1 = (224, 224)
    shape2 = (299, 299)
    model_1_path = 'model1.tflite'
    model_2_path = 'model2.tflite'

    class1 = predict(image_path, shape1, model_1_path)
    if class1 == 1:
        return 'n'

    class2 = predict(image_path, shape2, model_2_path)
    
    if class2 == 0:
        return 'b'
    else:
        return 'm'
    
class App(tk.Tk):
    '''
    This class is used to create the GUI application.

    '''
    def __init__(self) -> None:
        super().__init__()
        self.title("Multi-Stage Diagnosis")
        self.geometry("400x400")
        self.create_widgets()

    def create_widgets(self) -> None:
        '''
        This method is used to create the widgets of the application.

        '''
        self.upload_button = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=20)
        
        self.loading_label = tk.Label(self, text="", font=("Helvetica", 16))
        self.loading_label.pack(pady=20)
        
        self.output_text = tk.Text(self, wrap="word", height=10)
        self.output_text.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        self.output_text.config(state=tk.DISABLED)
        
    def display_loading(self) -> None:
        '''
        This method is used to display a loading message.
        
        '''
        self.loading_label.config(text="Loading...")
    
    def display_output(self, text: str) -> None:
        '''
        This method is used to display the output text.
        
        :param text: The output text
    
        '''
        self.loading_label.config(text="")
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.tag_add("center", "1.0", "end")
        self.output_text.tag_configure("center", justify="center")
        self.output_text.config(state=tk.DISABLED)

    def upload_image(self) -> None:
        '''
        This method is used to upload an image.
            
        '''

        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_loading()
            image_name = file_path.split('/')[-1]
            image_path = f'./Testing/{image_name}'
            threading.Thread(target=self.run_diagnosis, args=(image_path,)).start()
    
    def run_diagnosis(self, image_path: str) -> None:
        '''
        This method is used to run the diagnosis on a separate thread.

        :param image_path: The path to the image

        '''

        result = diagnose(image_path)
        res = ''
        if result == 'n':
            res = 'Normal'
        elif result == 'b':
            res = 'Benign'
        elif result == 'm':
            res = 'Malignant'
        else:
            res = 'Error'
            
        self.display_output(res)
                    
app = App()
app.mainloop()
