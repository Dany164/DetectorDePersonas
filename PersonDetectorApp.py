"""
Detector de Personas con Roboflow
---------------------------------
Este programa captura una imagen con la webcam y usa la API de Roboflow
para detectar personas y otros objetos en la imagen.

Requisitos:
- pip install roboflow opencv-python pillow
"""

import os
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tkinter as tk
from tkinter import messagebox, Label
from PIL import Image, ImageTk
import threading

class PersonDetectorApp:
    def __init__(self, root):
        """
        Inicializa la aplicación con una interfaz gráfica básica
        """
        self.root = root
        self.root.title("Detector de Personas")
        self.root.geometry("800x600")
        
        # Variables para almacenar resultados
        self.detection_results = []
        self.captured_image_path = None
        self.processed_image_path = None
        
        # Variables de configuración
        self.api_key = "gRjtKzAcootXzZo9s0nu"  # IMPORTANTE: Debes establecer tu API key de Roboflow aquí
        self.confidence_threshold = 40  # Umbral de confianza para detecciones
        
        # Inicializar Roboflow si la API key está configurada
        self.rf = None
        self.model = None
        self.setup_roboflow()
        
        # Crear interfaz
        self.create_widgets()
    
    def setup_roboflow(self):
        """
        Configura la conexión con Roboflow y carga el modelo de detección de personas
        """
        if not self.api_key:
            messagebox.showwarning("Configuración incompleta", 
                                 "Por favor configura tu API key de Roboflow en el código")
            return
        
        try:
            # IMPORTANTE: Inicializar Roboflow con la API key
            self.rf = Roboflow(api_key=self.api_key)
            
            # IMPORTANTE: Cargar un modelo preentrenado para detección de personas
            # El modelo "coco" es ampliamente usado y detecta personas y otros objetos comunes
            self.model = self.rf.workspace().project("my-first-project-sz6xo").version(1).model
            
            print("Modelo cargado correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"Error al conectar con Roboflow: {str(e)}")
            print(f"Error al conectar con Roboflow: {str(e)}")
    
    def create_widgets(self):
        """
        Crea los elementos de la interfaz gráfica
        """
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para controles
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Botón para capturar imagen
        self.capture_btn = tk.Button(controls_frame, text="Capturar Imagen", 
                                   command=self.capture_and_analyze, height=2, 
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.capture_btn.pack(side=tk.LEFT, padx=10)
        
        # Etiqueta de estado
        self.status_label = tk.Label(controls_frame, text="Listo para capturar", 
                                   font=("Arial", 10), bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Frame para imagen
        self.image_frame = tk.Frame(main_frame, bg="#EEEEEE", height=400)
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Etiqueta para mostrar imagen
        self.image_label = tk.Label(self.image_frame, bg="#EEEEEE")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Frame para resultados
        results_frame = tk.Frame(main_frame)
        results_frame.pack(fill=tk.X, pady=10)
        
        # Etiqueta para resultados
        self.results_label = tk.Label(results_frame, text="Resultados aparecerán aquí", 
                                    font=("Arial", 10), bg="#F0F0F0", anchor=tk.W, 
                                    justify=tk.LEFT, padx=10, pady=10)
        self.results_label.pack(fill=tk.X)
    
    def update_status(self, message, is_error=False):
        """
        Actualiza el mensaje de estado en la interfaz
        """
        self.status_label.config(text=message, fg="red" if is_error else "black")
        self.root.update()
    
    def capture_image(self):
        """
        Captura una imagen desde la webcam
        """
        self.update_status("Accediendo a la cámara...")
        
        try:
            # IMPORTANTE: Acceder a la webcam (0 es normalmente la cámara integrada)
            cap = cv2.VideoCapture(0)
            
            # Verificar si la cámara se abrió correctamente
            if not cap.isOpened():
                raise Exception("No se pudo acceder a la cámara")
            
            # Dar tiempo a la cámara para ajustarse
            time.sleep(1)
            
            # Capturar imagen
            ret, frame = cap.read()
            if not ret:
                raise Exception("Error al capturar imagen")
            
            # Generar nombre de archivo único con timestamp
            timestamp = int(time.time())
            filename = f"captured_image_{timestamp}.jpg"
            
            # Guardar imagen
            cv2.imwrite(filename, frame)
            
            # Liberar la cámara
            cap.release()
            
            return filename
            
        except Exception as e:
            raise Exception(f"Error al capturar imagen: {str(e)}")
    
    def analyze_image(self, image_path):
        """
        Analiza la imagen con el modelo de Roboflow
        """
        if not self.model:
            raise Exception("El modelo no está configurado correctamente")
        
        # IMPORTANTE: Predecir con el modelo de Roboflow
        # confidence: umbral de confianza mínima (0-100)
        # overlap: umbral para eliminación de detecciones duplicadas (0-100)
        results = self.model.predict(image_path, confidence=self.confidence_threshold, overlap=30)
        
        return results
    
    def draw_detections(self, image_path, predictions):
        """
        Dibuja cuadros delimitadores alrededor de los objetos detectados
        """
        # Cargar imagen con PIL
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Intentar cargar una fuente, si falla usar la predeterminada
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Dibujar cada predicción
        for prediction in predictions:
            # Extraer coordenadas y clase
            x1 = prediction['x'] - prediction['width'] / 2
            y1 = prediction['y'] - prediction['height'] / 2
            x2 = prediction['x'] + prediction['width'] / 2
            y2 = prediction['y'] + prediction['height'] / 2
            
            class_name = prediction['class']
            confidence = prediction['confidence']
            
            # Color especial para personas (verde)
            color = "green" if class_name.lower() == "person" else "red"
            
            # Dibujar rectángulo
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Dibujar etiqueta
            label_text = f"{class_name}: {confidence:.1f}%"
            draw.rectangle([x1, y1-25, x1+len(label_text)*9, y1], fill=color)
            draw.text((x1+5, y1-25), label_text, fill="white", font=font)
        
        # Guardar imagen procesada
        output_path = f"processed_{os.path.basename(image_path)}"
        image.save(output_path)
        
        return output_path
    
    def display_image(self, image_path):
        """
        Muestra la imagen en la interfaz
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Redimensionar manteniendo proporción
            image_width, image_height = image.size
            frame_width = self.image_frame.winfo_width()
            frame_height = self.image_frame.winfo_height()
            
            # Calcular nueva dimensión
            ratio = min(frame_width/image_width, frame_height/image_height)
            new_width = int(image_width * ratio)
            new_height = int(image_height * ratio)
            
            # Redimensionar
            image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Convertir para tkinter
            tk_image = ImageTk.PhotoImage(image)
            
            # Mostrar imagen
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # Mantener referencia
            
        except Exception as e:
            self.update_status(f"Error al mostrar imagen: {str(e)}", True)
    
    def format_results(self, predictions):
        """
        Da formato a los resultados para mostrarlos en la interfaz
        """
        # Contar personas
        person_count = sum(1 for p in predictions if p['class'].lower() == 'person')
        
        # Crear mensaje de resultados
        if person_count > 0:
            result_text = f"¡Se detectaron {person_count} personas en la imagen!\n\n"
        else:
            result_text = "No se detectaron personas en la imagen.\n\n"
        
        # Agrupar objetos por clase
        objects = {}
        for p in predictions:
            class_name = p['class']
            if class_name in objects:
                objects[class_name] += 1
            else:
                objects[class_name] = 1
        
        # Añadir lista de objetos detectados
        if objects:
            result_text += "Objetos detectados:\n"
            for obj, count in objects.items():
                result_text += f"- {obj}: {count}\n"
        
        return result_text
    
    def capture_and_analyze(self):
        """
        Función principal que captura y analiza la imagen
        """
        # Deshabilitar botón mientras se procesa
        self.capture_btn.config(state=tk.DISABLED)
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self._process_image_thread).start()
    
    def _process_image_thread(self):
        """
        Proceso de captura y análisis en un hilo separado
        """
        try:
            # Capturar imagen
            self.update_status("Capturando imagen...")
            self.captured_image_path = self.capture_image()
            
            # Mostrar imagen capturada
            self.update_status("Imagen capturada. Analizando...")
            self.root.after(0, lambda: self.display_image(self.captured_image_path))
            
            # Analizar imagen con Roboflow
            predictions = self.analyze_image(self.captured_image_path)
            self.detection_results = predictions
            
            # Dibujar detecciones en la imagen
            self.processed_image_path = self.draw_detections(self.captured_image_path, predictions)
            
            # Mostrar imagen procesada
            self.root.after(0, lambda: self.display_image(self.processed_image_path))
            
            # Actualizar resultados
            results_text = self.format_results(predictions)
            self.root.after(0, lambda: self.results_label.config(text=results_text))
            
            # Actualizar estado
            self.update_status("Análisis completado")
            
        except Exception as e:
            self.root.after(0, lambda: self.update_status(f"Error: {str(e)}", True))
        finally:
            # Rehabilitar botón
            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))

def main():
    # Crear ventana de tkinter
    root = tk.Tk()
    
    # Crear instancia de la aplicación
    app = PersonDetectorApp(root)
    
    # Iniciar bucle principal
    root.mainloop()

if __name__ == "__main__":
    main()