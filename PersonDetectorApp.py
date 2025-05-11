"""
Detector de Personas con Roboflow - Versión Mejorada
----------------------------------------------------
Este programa captura una imagen con la webcam y usa la API de Roboflow
para detectar personas y otros objetos en la imagen, con optimizaciones
para mejorar la precisión en la detección de personas.

Requisitos:
- pip install roboflow opencv-python pillow numpy
"""

import os
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import tkinter as tk
from tkinter import messagebox, Label, Scale, IntVar, BooleanVar, Checkbutton
from PIL import Image, ImageTk
import threading
import json
import logging

# Configurar logging para depuración
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("detector_log.txt"), logging.StreamHandler()]
)
logger = logging.getLogger("PersonDetector")

class PersonDetectorApp:
    def __init__(self, root):
        """
        Inicializa la aplicación con una interfaz gráfica mejorada
        """
        self.root = root
        self.root.title("Detector de Personas Mejorado")
        self.root.geometry("900x700")
        
        # Variables para almacenar resultados
        self.detection_results = []
        self.captured_image_path = None
        self.processed_image_path = None
        self.raw_api_response = None
        
        # Variables de configuración con valores más óptimos
        self.api_key = ""  # La misma API key del original
        
        # Reducir el umbral de confianza para aumentar la sensibilidad
        self.confidence_threshold = 25  # Valor reducido (antes era 40)
        self.overlap_threshold = 25     # Valor reducido para NMS
        
        # Inicializar variables para configuración de la interfaz
        self.confidence_value = IntVar(value=self.confidence_threshold)
        self.preprocessing_enabled = BooleanVar(value=True)
        self.debug_mode = BooleanVar(value=True)
        self.use_local_detection = BooleanVar(value=False)
        
        # Inicializar Roboflow
        self.rf = None
        self.model = None
        self.setup_roboflow()
        
        # Crear interfaz de usuario mejorada
        self.create_widgets()
        
        # Cargar modelo de Haar Cascade para detección local de respaldo
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        
        logger.info("Aplicación inicializada correctamente")
    
    def setup_roboflow(self):
        """
        Configura la conexión con Roboflow y carga el modelo de detección de personas
        """
        if not self.api_key:
            messagebox.showwarning("Configuración incompleta", 
                                 "Por favor configura tu API key de Roboflow en el código")
            return
        
        try:
            # Inicializar Roboflow con la API key
            self.rf = Roboflow(api_key=self.api_key)
            
            # Cargar modelo para detección de personas
            # Intentar utilizar un modelo más robusto (YOLO) si está disponible
            try:
                # Primero intentamos con el modelo original
                self.model = self.rf.workspace().project("").version(3).model
                logger.info("Modelo original cargado correctamente")
            except Exception as e:
                logger.warning(f"No se pudo cargar el modelo original: {str(e)}")
                # Si falla, intenta con un modelo COCO preentrenado que es bueno detectando personas
                try:
                    self.model = self.rf.workspace().project("coco").version(1).model
                    logger.info("Modelo COCO cargado como alternativa")
                except:
                    logger.error("No se pudo cargar ningún modelo alternativo")
                    raise
            
            logger.info("Modelo cargado correctamente")
        except Exception as e:
            error_msg = f"Error al conectar con Roboflow: {str(e)}"
            logger.error(error_msg)
            messagebox.showerror("Error", error_msg)
    
    def create_widgets(self):
        """
        Crea los elementos de la interfaz gráfica mejorada
        """
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame para configuración
        config_frame = tk.Frame(main_frame, relief=tk.RIDGE, bd=2)
        config_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Control deslizante para umbral de confianza
        tk.Label(config_frame, text="Umbral de confianza:").grid(row=0, column=0, padx=5, pady=5)
        confidence_slider = Scale(config_frame, from_=10, to=90, orient=tk.HORIZONTAL, 
                                variable=self.confidence_value, length=200)
        confidence_slider.grid(row=0, column=1, padx=5, pady=5)
        
        # Checkbox para preprocesamiento de imagen
        preprocess_check = Checkbutton(config_frame, text="Preprocesar imagen", 
                                     variable=self.preprocessing_enabled)
        preprocess_check.grid(row=0, column=2, padx=10, pady=5)
        
        # Checkbox para modo de depuración
        debug_check = Checkbutton(config_frame, text="Modo depuración", 
                                variable=self.debug_mode)
        debug_check.grid(row=0, column=3, padx=10, pady=5)
        
        # Checkbox para detección local (fallback)
        local_check = Checkbutton(config_frame, text="Usar detección local", 
                               variable=self.use_local_detection)
        local_check.grid(row=0, column=4, padx=10, pady=5)
        
        # Frame para controles
        controls_frame = tk.Frame(main_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        # Botón para capturar imagen (más grande y llamativo)
        self.capture_btn = tk.Button(controls_frame, text="Capturar y Analizar", 
                                   command=self.capture_and_analyze, height=2, width=20,
                                   bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        self.capture_btn.pack(side=tk.LEFT, padx=10)
        
        # Botón para cargar una imagen de prueba (útil para debugging)
        self.test_btn = tk.Button(controls_frame, text="Usar Imagen de Test", 
                                command=self.use_test_image, height=2,
                                bg="#2196F3", fg="white", font=("Arial", 12))
        self.test_btn.pack(side=tk.LEFT, padx=10)
        
        # Etiqueta de estado mejorada
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
        results_frame = tk.Frame(main_frame, relief=tk.RIDGE, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Etiqueta para resultados
        self.results_label = tk.Label(results_frame, text="Resultados aparecerán aquí", 
                                    font=("Arial", 10), bg="#F0F0F0", anchor=tk.NW, 
                                    justify=tk.LEFT, padx=10, pady=10)
        self.results_label.pack(fill=tk.BOTH, expand=True)
        
        # Etiqueta para información de depuración
        self.debug_label = tk.Label(main_frame, text="", font=("Courier", 8), 
                                  bg="#F5F5F5", anchor=tk.W, justify=tk.LEFT, 
                                  padx=5, pady=5, relief=tk.SUNKEN)
        self.debug_label.pack(fill=tk.X, pady=5)
    
    def update_status(self, message, is_error=False):
        """
        Actualiza el mensaje de estado en la interfaz
        """
        self.status_label.config(text=message, fg="red" if is_error else "black")
        self.root.update()
        
        # Registrar en el log
        if is_error:
            logger.error(message)
        else:
            logger.info(message)
    
    def update_debug_info(self, message):
        """
        Actualiza la información de depuración en la interfaz
        """
        if self.debug_mode.get():
            self.debug_label.config(text=message)
    
    def preprocess_image(self, image_path):
        """
        Preprocesa la imagen para mejorar la detección
        """
        if not self.preprocessing_enabled.get():
            return image_path
            
        try:
            # Leer imagen con OpenCV
            img = cv2.imread(image_path)
            
            # Verificar si la imagen se cargó correctamente
            if img is None:
                logger.error(f"No se pudo cargar la imagen desde {image_path}")
                return image_path
                
            # Convertir a escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar ecualización de histograma para mejorar el contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Reducir ruido
            denoised = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Convertir de nuevo a color
            enhanced = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            
            # Ajustar brillo y contraste
            alpha = 1.2  # Contraste
            beta = 10    # Brillo
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            # Combinar con la original con peso
            result = cv2.addWeighted(img, 0.7, enhanced, 0.3, 0)
            
            # Guardar imagen preprocesada
            preprocessed_path = f"preprocessed_{os.path.basename(image_path)}"
            cv2.imwrite(preprocessed_path, result)
            
            logger.info(f"Imagen preprocesada guardada como {preprocessed_path}")
            return preprocessed_path
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {str(e)}")
            return image_path  # En caso de error, devolver la imagen original
    
    def capture_image(self):
        """
        Captura una imagen desde la webcam con mejor calidad
        """
        self.update_status("Accediendo a la cámara...")
        
        try:
            # Acceder a la webcam
            cap = cv2.VideoCapture(0)
            
            # Configurar resolución más alta
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Verificar si la cámara se abrió correctamente
            if not cap.isOpened():
                raise Exception("No se pudo acceder a la cámara")
            
            # Dar tiempo a la cámara para ajustarse
            time.sleep(1.5)  # Aumentado para mejor ajuste
            
            # Capturar múltiples frames y quedarse con el mejor
            frames = []
            for _ in range(5):  # Tomar 5 frames
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                time.sleep(0.1)
            
            # Liberar la cámara
            cap.release()
            
            if not frames:
                raise Exception("Error al capturar imágenes")
            
            # Seleccionar el frame menos borroso (con mayor varianza Laplaciana)
            best_frame = max(frames, key=self.variance_of_laplacian)
            
            # Generar nombre de archivo único con timestamp
            timestamp = int(time.time())
            filename = f"captured_image_{timestamp}.jpg"
            
            # Guardar imagen con mejor calidad
            cv2.imwrite(filename, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            logger.info(f"Imagen capturada y guardada como {filename}")
            return filename
            
        except Exception as e:
            error_msg = f"Error al capturar imagen: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def variance_of_laplacian(self, image):
        """
        Calcula la varianza del Laplaciano para determinar el enfoque de la imagen.
        Un valor más alto indica una imagen más nítida.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def use_test_image(self):
        """
        Usa una imagen de prueba para depuración (crea una si no existe)
        """
        test_image_path = "test_image.jpg"
        
        # Si no existe la imagen de prueba, usar la webcam para crear una
        if not os.path.exists(test_image_path):
            self.update_status("Creando imagen de prueba...")
            try:
                test_image_path = self.capture_image()
                os.rename(test_image_path, "test_image.jpg")
                test_image_path = "test_image.jpg"
            except:
                self.update_status("No se pudo crear imagen de prueba", True)
                return
        
        # Procesar la imagen de prueba
        self.captured_image_path = test_image_path
        self.update_status("Usando imagen de prueba para análisis")
        
        # Procesar en un hilo separado
        threading.Thread(target=self._process_image_thread, 
                       args=(test_image_path,)).start()
    
    def analyze_image(self, image_path):
        """
        Analiza la imagen con el modelo de Roboflow o detección local
        """
        # Actualizar umbral de confianza desde la interfaz
        self.confidence_threshold = self.confidence_value.get()
        
        if self.use_local_detection.get():
            return self.local_detection(image_path)
        
        if not self.model:
            raise Exception("El modelo no está configurado correctamente")
        
        try:
            # Predecir con el modelo de Roboflow
            results = self.model.predict(image_path, confidence=self.confidence_threshold, 
                                         overlap=self.overlap_threshold)
            
            # Guardar respuesta completa para depuración
            self.raw_api_response = results.json
            
            # Mostrar respuesta cruda para depuración
            debug_text = f"API Response: {json.dumps(results.json, indent=2)[:500]}..."
            logger.debug(debug_text)
            self.update_debug_info(debug_text)
            
            return results
            
        except Exception as e:
            error_msg = f"Error al analizar con Roboflow: {str(e)}"
            logger.error(error_msg)
            
            # Si falla Roboflow, intentar con detección local
            self.update_status("Fallback: usando detección local...")
            return self.local_detection(image_path)
    
    def local_detection(self, image_path):
        """
        Método alternativo usando OpenCV para detección local de personas
        """
        try:
            # Leer imagen
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            # Detectar caras
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Detectar cuerpos
            bodies = self.body_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(80, 200))
            
            # Crear estructura similar a la respuesta de Roboflow
            results = {"predictions": []}
            
            # Añadir caras detectadas
            for (x, y, w, h) in faces:
                confidence = 80  # Valor arbitrario alto para caras
                results["predictions"].append({
                    "x": x + w//2,
                    "y": y + h//2,
                    "width": w,
                    "height": h,
                    "class": "person",
                    "confidence": confidence
                })
            
            # Añadir cuerpos detectados
            for (x, y, w, h) in bodies:
                confidence = 70  # Valor arbitrario para cuerpos
                results["predictions"].append({
                    "x": x + w//2,
                    "y": y + h//2,
                    "width": w,
                    "height": h,
                    "class": "person",
                    "confidence": confidence
                })
            
            # Crear un objeto similar al devuelto por Roboflow
            class LocalDetectionResults:
                def __init__(self, results_dict):
                    self.json = results_dict
                    self.predictions = results_dict["predictions"]
                
                def __iter__(self):
                    return iter(self.predictions)
                
                def __len__(self):
                    return len(self.predictions)
            
            logger.info(f"Detección local: {len(results['predictions'])} personas encontradas")
            return LocalDetectionResults(results)
            
        except Exception as e:
            error_msg = f"Error en detección local: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Error en detección local: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def draw_detections(self, image_path, predictions):
        """
        Dibuja cuadros delimitadores mejorados alrededor de los objetos detectados
        """
        # Cargar imagen con PIL
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Intentar cargar una fuente, si falla usar la predeterminada
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            try:
                # Intentar con otras fuentes comunes
                font = ImageFont.truetype("DejaVuSans.ttf", 20)
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
            
            # Color basado en la clase y confianza
            if class_name.lower() == "person":
                # Gradiente de color basado en confianza para personas
                # Verde más intenso para mayor confianza
                if confidence > 80:
                    color = "#00BB00"  # Verde intenso
                elif confidence > 60:
                    color = "#44DD00"  # Verde claro
                elif confidence > 40:
                    color = "#BBDD00"  # Verde amarillento
                else:
                    color = "#DDCC00"  # Amarillo
            else:
                # Otros objetos en azul con intensidad según confianza
                intensity = int(155 + min(confidence, 100))
                color = f"#{0:02x}{0:02x}{intensity:02x}"  # Azul
            
            # Dibujar rectángulo con borde más visible
            for i in range(3):  # Dibujar varios rectángulos para efecto de grosor
                offset = i * 1
                draw.rectangle([
                    x1-offset, y1-offset, x2+offset, y2+offset
                ], outline=color, width=1)
            
            # Añadir fondo semitransparente para la etiqueta
            label_text = f"{class_name}: {confidence:.1f}%"
            text_width = len(label_text) * 10
            text_height = 25
            
            # Rectángulo semitransparente para el texto
            draw_overlay = ImageDraw.Draw(image, 'RGBA')
            draw_overlay.rectangle(
                [x1, y1-text_height, x1+text_width, y1],
                fill=(0, 0, 0, 160)  # Negro semitransparente
            )
            
            # Dibujar etiqueta
            draw.text((x1+5, y1-text_height+2), label_text, fill="white", font=font)
        
        # Añadir información en la parte inferior
        info_text = f"Detecciones: {len(predictions)} | Umbral: {self.confidence_threshold}%"
        draw_overlay = ImageDraw.Draw(image, 'RGBA')
        draw_overlay.rectangle(
            [0, image.height-30, image.width, image.height],
            fill=(0, 0, 0, 160)  # Negro semitransparente
        )
        draw.text((10, image.height-25), info_text, fill="white", font=font)
        
        # Guardar imagen procesada
        output_path = f"processed_{os.path.basename(image_path)}"
        image.save(output_path)
        
        logger.info(f"Imagen con detecciones guardada como {output_path}")
        return output_path
    
    def display_image(self, image_path):
        """
        Muestra la imagen en la interfaz con mejor ajuste
        """
        try:
            # Cargar imagen
            image = Image.open(image_path)
            
            # Esperar a que el widget tenga tamaño
            self.root.update_idletasks()
            
            # Redimensionar manteniendo proporción
            image_width, image_height = image.size
            frame_width = self.image_frame.winfo_width() or 700
            frame_height = self.image_frame.winfo_height() or 400
            
            # Calcular nueva dimensión
            ratio = min(frame_width/image_width, frame_height/image_height)
            new_width = int(image_width * ratio)
            new_height = int(image_height * ratio)
            
            # Redimensionar con el mejor método disponible
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convertir para tkinter
            tk_image = ImageTk.PhotoImage(image)
            
            # Mostrar imagen
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # Mantener referencia
            
        except Exception as e:
            error_msg = f"Error al mostrar imagen: {str(e)}"
            self.update_status(error_msg, True)
    
    def format_results(self, predictions):
        """
        Da formato a los resultados para mostrarlos en la interfaz
        """
        # Contar personas y clasificar por confianza
        persons = [p for p in predictions if p['class'].lower() == 'person']
        high_conf_persons = [p for p in persons if p['confidence'] >= 60]
        medium_conf_persons = [p for p in persons if 40 <= p['confidence'] < 60]
        low_conf_persons = [p for p in persons if p['confidence'] < 40]
        
        # Crear mensaje de resultados con más detalles
        if persons:
            result_text = f"¡DETECCIÓN EXITOSA!\n\n"
            result_text += f"Se detectaron {len(persons)} personas en la imagen:\n"
            result_text += f"- Alta confianza (>60%): {len(high_conf_persons)}\n"
            result_text += f"- Media confianza (40-60%): {len(medium_conf_persons)}\n"
            result_text += f"- Baja confianza (<40%): {len(low_conf_persons)}\n\n"
        else:
            result_text = "No se detectaron personas en la imagen.\n\n"
            result_text += "Prueba a:\n"
            result_text += "- Reducir el umbral de confianza\n"
            result_text += "- Mejorar la iluminación\n"
            result_text += "- Activar el preprocesamiento de imagen\n"
            result_text += "- Activar la detección local\n\n"
        
        # Agrupar objetos por clase con más detalle
        objects = {}
        for p in predictions:
            class_name = p['class']
            confidence = p['confidence']
            
            if class_name not in objects:
                objects[class_name] = {"count": 1, "avg_conf": confidence}
            else:
                current = objects[class_name]
                new_count = current["count"] + 1
                new_avg = (current["avg_conf"] * current["count"] + confidence) / new_count
                objects[class_name] = {"count": new_count, "avg_conf": new_avg}
        
        # Añadir lista de objetos detectados
        if objects:
            result_text += "Objetos detectados:\n"
            for obj, data in sorted(objects.items(), key=lambda x: x[1]["count"], reverse=True):
                result_text += f"- {obj}: {data['count']} (confianza media: {data['avg_conf']:.1f}%)\n"
        
        # Añadir sugerencias si no hay detecciones o son pocas
        if not persons:
            result_text += "\nSugerencias para mejorar la detección:\n"
            result_text += "1. Asegúrate de estar dentro del campo de visión de la cámara\n"
            result_text += "2. Mejora la iluminación (evita contraluz)\n"
            result_text += "3. Reduce el umbral de confianza usando el deslizador\n"
            result_text += "4. Prueba con la detección local activando la casilla\n"
        
        return result_text
    
    def capture_and_analyze(self):
        """
        Función principal que captura y analiza la imagen
        """
        # Deshabilitar botón mientras se procesa
        self.capture_btn.config(state=tk.DISABLED)
        self.test_btn.config(state=tk.DISABLED)
        
        # Ejecutar en un hilo separado para no bloquear la interfaz
        threading.Thread(target=self._process_image_thread).start()
    
    def _process_image_thread(self, image_path=None):
        """
        Proceso de captura y análisis en un hilo separado
        """
        try:
            if not image_path:
                # Capturar imagen
                self.update_status("Capturando imagen...")
                self.captured_image_path = self.capture_image()
                image_path = self.captured_image_path
            
            # Mostrar imagen capturada
            self.update_status("Imagen capturada. Preprocesando...")
            self.root.after(0, lambda: self.display_image(image_path))
            
            # Preprocesar imagen para mejor detección
            preprocessed_path = self.preprocess_image(image_path)
            
            # Analizar imagen
            self.update_status("Analizando imagen...")
            predictions = self.analyze_image(preprocessed_path)
            self.detection_results = predictions
            
            # Dibujar detecciones en la imagen
            self.update_status("Procesando resultados...")
            self.processed_image_path = self.draw_detections(
                preprocessed_path if self.preprocessing_enabled.get() else image_path, 
                predictions
            )
            
            # Mostrar imagen procesada
            self.root.after(0, lambda: self.display_image(self.processed_image_path))
            
            # Actualizar resultados
            results_text = self.format_results(predictions)
            self.root.after(0, lambda: self.results_label.config(text=results_text))
            
            # Actualizar estado
            person_count = sum(1 for p in predictions if p['class'].lower() == 'person')
            if person_count > 0:
                status_msg = f"¡Análisis completado! Se detectaron {person_count} personas"
                self.update_status(status_msg)
            else:
                self.update_status("Análisis completado. No se detectaron personas.")
            
        except Exception as e:
            error_msg = f"Error durante el procesamiento: {str(e)}"
            logger.error(error_msg)
            self.root.after(0, lambda: self.update_status(error_msg, True))
        finally:
            # Rehabilitar botones
            self.root.after(0, lambda: self.capture_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.test_btn.config(state=tk.NORMAL))

def main():
    # Crear ventana de tkinter
    root = tk.Tk()
    
    # Establecer icono si está disponible
    try:
        icon_path = "app_icon.ico"
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
    except:
        pass  # Ignorar si no se puede establecer el icono
    
    # Crear instancia de la aplicación
    app = PersonDetectorApp(root)
    
    # Mostrar mensaje de inicio
    messagebox.showinfo("Detector de Personas", 
                      "Esta versión incluye mejoras para la detección de personas:\n\n"
                      "- Umbral de confianza reducido\n"
                      "- Preprocesamiento de imagen\n"
                      "- Detección local como alternativa\n"
                      "- Botón de prueba con imagen estática\n\n"
                      "Si tienes problemas con la detección, prueba a reducir el umbral "
                      "de confianza con el deslizador o activa la detección local.")
    
    # Iniciar bucle principal
    root.mainloop()

if __name__ == "__main__":
    main()