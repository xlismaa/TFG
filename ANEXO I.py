import cv2
import os

# Ruta de la carpeta que contiene los vídeos
video_folder = 'C:/Users/ismae/Desktop/TFG/Vídeos Ecografías/Ecografías Fascia'
# Carpeta donde se guardarán los frames
output_folder = 'frames'

# Crear la carpeta si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Listar todos los archivos en la carpeta
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    
    # Abrir el vídeo
    cap = cv2.VideoCapture(video_path)

    # Comprobar si el vídeo se abrió correctamente
    if not cap.isOpened():
        print(f"Error al abrir el vídeo: {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)  # Obtener el número de frames por segundo
    frame_interval = int(fps / 5)  # Calcular cuántos frames ignorar

    frame_count = 0
    saved_count = 0  # Contador para frames guardados

    # Extraer frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Guardar el frame con el nombre del vídeo
            frame_filename = os.path.join(output_folder, f'{os.path.splitext(video_file)[0]}_frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f'Saved frame {saved_count} from {video_file}')
            saved_count += 1

        frame_count += 1

    # Liberar el objeto de captura
    cap.release()
    print(f'Se extrajeron {saved_count} frames de {video_file}.')