import streamlit as st
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from torchvision import transforms
from streamlit_option_menu import option_menu
import os

# Cargar el modelo y el procesador de imágenes
model_path = '/content/drive/MyDrive/PT/trained_model'
model = AutoModelForImageClassification.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

# Definir las transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

# Función para predecir la emoción en una imagen
def predict_emotion(image):
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        top_probs, top_labels = torch.topk(probs, probs.size(1), dim=1)
    predictions = [{'score': score.item(), 'label': model.config.id2label[label.item()]}
                   for score, label in zip(top_probs[0], top_labels[0])]
    return predictions

# Inicializar el estado de la aplicación
if 'menu_option' not in st.session_state:
    st.session_state.menu_option = "Principal"
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None  # Variable para almacenar la imagen cargada

# Interfaz de Streamlit
st.set_page_config(page_title="Detección de emociones", layout="centered")

with st.sidebar:
    st.header("Bienvenid@ a mi aplicación.")
    st.write("""
    Esta es una aplicación para la detección de emociones, diseñada para identificar las diferentes
    emociones que una persona puede expresar. Las imágenes no solo capturan momentos, también son
    poderosas fuentes de emociones. Un solo gesto puede revelar una historia, y esta herramienta está
    aquí para ayudarte a descifrarla.
    ¡Explora y descubre las emociones a través de tus imágenes, porque a veces una imagen dice más que mil palabras!""")
    st.write("---")
    st.write("")
    st.write("")
    st.write("")
    st.write("Si necesitas ayuda presiona el boton😀")
    if st.sidebar.button("Ayuda"):
      st.sidebar.markdown("Video de ayuda: [Haz clic aquí](https://www.youtube.com/watch?v=zeS2FlxF_0s&t=1702s)")

# Agregar logos en la parte superior
logo1 = Image.open('/content/drive/MyDrive/PT/Imagenes_Streamlit/logo.png')
logo2 = Image.open('/content/drive/MyDrive/PT/Imagenes_Streamlit/UAM.png')

# Contenedor para los logos
logo_container = st.container()
with logo_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.image(logo1, width=100)
    with col2:
        st.write("")
    with col3:
        st.image(logo2, width=400)

# Menú de opciones en la parte superior
st.session_state.menu_option = option_menu("___________________Menu___________________",
                                             [ "Conoce", "Principal",  "Créditos"],
                                             icons=[ 'question-circle','house',  'person-fill'],
                                             menu_icon='three-dots',
                                             default_index=1,
                                             orientation='horizontal')

# Define un diccionario de emociones y sus emojis
emotion_emojis = {
    "Feliz": "😊",
    "Triste": "😢",
    "Sorpresa": "😮",
    "Enojo": "😡",
    "Miedo": "😱",
    "Desagrado": "😖",
    "Neutral": "😐"
}

#Opcion de principal y componentes
if st.session_state.menu_option == "Principal":

    st.header("Detección de Emociones en Imágenes")
    st.write("---")
    st.write("😲 ¡Descubre la emoción que tu imagen puede revelar! 😄📸")
    st.write("""Esta aplicación está diseñada para analizar imágenes que muestren un solo rostro.
    Asegúrate de que la imagen destaque una única cara para obtener los mejores resultados.""")

    # Opción para cargar una imagen desde la PC
    st.subheader("Cargar imagen desde mi PC")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    col1, col2 = st.columns([2, 1])
    col2.subheader("Tomar una foto")
    camera_photo = col2.camera_input(" ", label_visibility="collapsed")

    # Revisar si hay un archivo subido o una foto tomada
    image = None

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            col2.success("La imagen fue cargada correctamente.")
            st.session_state.uploaded_image = image  # Almacena la imagen cargada
            st.session_state.selected_image = None  # Limpiar selección de imagen de prueba
        except Exception:
            col2.error("⚠️Error al cargar la imagen 😥. Por favor asegúrate de que el archivo sea válido 🧐.")
    elif camera_photo is not None:
        try:
            image = Image.open(camera_photo).convert('RGB')
            col2.success("La foto fue tomada correctamente.")
            st.session_state.uploaded_image = image  # Almacena la foto tomada
            st.session_state.selected_image = None  # Limpiar selección de imagen de prueba
        except Exception:
            col2.error("Error al tomar la foto.")

    # Galería de imágenes de prueba
    st.write("### Imágenes para probar")
    test_images_folder = '/content/drive/MyDrive/PT/Imagenes_Streamlit/Prueba_pagina'
    image_files = [f for f in os.listdir(test_images_folder) if f.endswith(('jpg', 'jpeg', 'png'))]

    num_images_per_row = 5
    num_rows = (len(image_files) + num_images_per_row - 1) // num_images_per_row

    for row in range(num_rows):
        cols = st.columns(num_images_per_row)
        for col in range(num_images_per_row):
            idx = row * num_images_per_row + col
            if idx < len(image_files):
                img_file = image_files[idx]
                img_path = os.path.join(test_images_folder, img_file)
                test_image = Image.open(img_path).convert('RGB')
                with cols[col]:
                    st.image(test_image, width=100)
                    if st.button(f"Probar", key=img_file):
                        st.session_state.selected_image = img_file
                        st.session_state.uploaded_image = None

    # Si hay una imagen disponible (cargada o seleccionada de la galería), mostrarla a la derecha y las probabilidades a la izquierda
    if image is not None or st.session_state.selected_image is not None:
        image_to_classify = image if image is not None else Image.open(os.path.join(test_images_folder, st.session_state.selected_image)).convert('RGB')
        col2.image(image_to_classify, caption='Imagen Cargada', width=300)
        with col1:
            st.write("Clasificando...")
            predictions = predict_emotion(image_to_classify)

            # Muestra la emoción detectada con su emoji
            detected_emotion = predictions[0]['label']
            emoji = emotion_emojis.get(detected_emotion, "❓")  # Usa un emoji de pregunta si no se encuentra la emoción
            st.write("Emoción detectada:")
            st.write(f"**{detected_emotion} {emoji}**")

            st.write("Probabilidades:")
            for pred in predictions:
                st.progress(pred['score'])
                emoji = emotion_emojis.get(pred['label'], "❓")  # Usa un emoji de pregunta si no se encuentra la emoción
                st.write(f"{pred['label']} {emoji}: {int(pred['score'] * 100)}%")

# Opción de conoce y sus componentes
elif st.session_state.menu_option == "Conoce":
    st.header("Transformadores Visuales y Reconocimiento de Emociones")
    st.write("### Transformadores Visuales (Vision Transformers - ViT)")
    st.write("""Los transformadores visuales (ViT) son un avance reciente en el campo de la visión por computadora.
    Basados en la arquitectura de transformadores utilizada originalmente para procesamiento de lenguaje natural,
    los ViT dividen las imágenes en 'parches' y procesan estos parches de manera similar a cómo los
    transformadores manejan secuencias de texto. Esto permite que el modelo aprenda relaciones espaciales
    complejas entre diferentes partes de la imagen, resultando en una comprensión más profunda del contenido.""")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image('/content/drive/MyDrive/PT/Imagenes_Streamlit/conoce1.png',
                 caption='Arquitectura de Vision Transformers',
                 width=400)

    # Segunda sección: Aplicaciones de los Modelos de Vision Transformers
    st.write("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Aplicaciones de los Modelos de Vision Transformers")
        st.write("""El uso de Vision Transformers en el reconocimiento de emociones es
        particularmente beneficioso, ya que pueden capturar detalles sutiles en las
        expresiones faciales humanas. Esto les permite identificar emociones con mayor
        precisión en situaciones complejas, como en imágenes con múltiples personas o
        con expresiones faciales mezcladas. Su capacidad para aprender patrones complejos
        a partir de grandes cantidades de datos facilita su uso en tareas como el análisis
        de emociones en tiempo real.""")
    with col2:
        st.image('/content/drive/MyDrive/PT/Imagenes_Streamlit/aplicacion.jpg',
                 caption='ViT en acción detectando emociones',
                 width=300)

    # Tercera sección: ViT para el Reconocimiento de Emociones
    st.write("---")
    col1, col2 = st.columns([1, 2])
    with col2:
        st.write("### Modelo ViT para el Reconocimiento de Emociones")
    col1, spacer, col2 = st.columns([1, 0.5, 2])
    with col1:
        st.image('/content/drive/MyDrive/PT/Imagenes_Streamlit/conoce2.png',
                 caption='Modelo ViT para emociones',
                 width=250)
    with col2:
        st.write("""En nuestra aplicación, utilizamos un modelo de Vision Transformers
        entrenado específicamente para la clasificación de emociones en imágenes.
        Este modelo ha sido ajustado con un conjunto de datos robusto que incluye
        expresiones faciales diversas. Gracias a su arquitectura, el modelo puede
        distinguir entre emociones como felicidad, tristeza, sorpresa y más con
        alta precisión, lo que lo convierte en una herramienta valiosa para
        aplicaciones en psicología, atención al cliente, y más.""")

elif st.session_state.menu_option == "Créditos":
    st.header("Créditos")
    st.write("Desarrollado por: Sebastian Hernández Mejía")
    st.write("Alumno de la carrera Licenciatura en Ingeniería en Computación")
    st.write("""Basado en 'Mothercreater/vit-Facial-Expression-Recognition', este modelo fue ajustado finamente con un conjunto de datos
    adicional que incluye imágenes de rostros en diversas expresiones emocionales.""")
    st.write("Datos: Se utilizó un conjunto de imágenes de AffectNet para el entrenamiento.")
    st.write("Agradecimientos especiales a mis asesores:")
    st.write("- Dra. Silvia Beatriz González Brambila")
    st.write("- M. en C. Josué Figueroa González")
