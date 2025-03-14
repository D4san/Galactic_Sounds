# app.py
from flask import Flask, request, render_template, send_file, redirect, url_for
import numpy as np
import os
import tempfile
import zipfile
from io import BytesIO
import json
import base64
from werkzeug.utils import secure_filename

from microtonal import (
    load_linear_spectrum,
    analizar_espectro,
    align_spectrum,
    map_spectrum_to_full_scale,
    map_spectrum_to_sequential_scale_full,
    create_midi_file_otoman_simplified,
    split_note_octave_microtonal_otoman_simplified,
    note_to_midi_microtonal_otoman_simplified,
    filtrar_espectro_por_rango
)

app = Flask(__name__)

def get_midi_sequence(note_list):
    """
    Convierte una lista de notas (en formato string) en una secuencia de números MIDI.
    Para 'silence' o notas no válidas, devuelve None.
    """
    midi_seq = []
    for note in note_list:
        if note == 'silence':
            midi_seq.append(None)
        else:
            info = split_note_octave_microtonal_otoman_simplified(note)
            if info:
                midi_num, _ = note_to_midi_microtonal_otoman_simplified(*info)
                midi_seq.append(midi_num)
            else:
                midi_seq.append(None)
    return midi_seq

@app.route('/')
def index():
    # Eliminar todos los archivos de la carpeta 'uploads'
    uploads_dir = os.path.join(app.root_path, 'uploads')
    if os.path.exists(uploads_dir):
        for f in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    return render_template('index.html')


@app.route('/preview', methods=['GET', 'POST'])
def preview():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    archivo = request.files.get('archivo')
    nombre_obra = request.form.get('nombre', 'MiObra').strip()
    min_wl = request.form.get('min_wl')
    max_wl = request.form.get('max_wl')
    file_path = request.form.get('temp_file')
    
    # Si no se recibe archivo y no hay file_path, error
    if not archivo and not file_path:
        return "No se recibió archivo", 400
    
    # Si se envía un nuevo archivo, guárdalo en la carpeta 'uploads'
    if archivo:
        uploads_dir = os.path.join(app.root_path, 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        filename = secure_filename(archivo.filename)
        file_path = os.path.join(uploads_dir, filename)
        archivo.save(file_path)
    
    # Verificar que el archivo exista
    if not os.path.exists(file_path):
        return "Archivo no encontrado", 400

    # Procesar el espectro
    wavelength, flux = load_linear_spectrum(file_path)
    resultados = analizar_espectro(wavelength, flux, region_continuo=(5000, 5500), fig=False)
    
    # Procesar valores de rango si se especificaron
    aplicar_rango = False
    rango_valido = False
    if min_wl and max_wl:
        try:
            min_wl = float(min_wl)
            max_wl = float(max_wl)
            if min_wl >= max_wl:
                print(f"Rango inválido: min_wl={min_wl}, max_wl={max_wl}")
            else:
                # Aseguramos que el rango no exceda el espectro original
                wl_min, wl_max = min(wavelength), max(wavelength)
                if min_wl < wl_min:
                    min_wl = wl_min
                if max_wl > wl_max:
                    max_wl = wl_max
                aplicar_rango = True
                rango_valido = True
        except ValueError as e:
            print(f"Error al convertir valores de rango: {e}")
    
    # Si se indicó un rango válido, filtramos los datos del espectro
    if aplicar_rango and rango_valido:
        # Se pasa el diccionario completo y el rango como tupla
        resultados = filtrar_espectro_por_rango(resultados, (min_wl, max_wl))
        resultados["es_filtrado"] = True
    else:
        resultados["es_filtrado"] = False

    # Extraer datos para la visualización y reproducción MIDI
    emission_wl, emission_flux = resultados["emision"]
    absorption_wl, absorption_flux = resultados["absorcion"]
    central_val = resultados["media_continuo"]
    max_val = np.max(resultados["full_spectrum"][1]) if len(resultados["full_spectrum"][1]) > 0 else 1.0
    full_wavelengths = resultados["full_spectrum"][0]
    
    # Alinear emisiones y absorciones
    aligned_emission = align_spectrum(full_wavelengths, emission_wl, emission_flux, central_val)
    aligned_absorption = align_spectrum(full_wavelengths, absorption_wl, absorption_flux, central_val)
    
    # Mapear a notas y generar secuencias MIDI
    _, mapped_emission = map_spectrum_to_sequential_scale_full(aligned_emission, max_val, central_val)
    _, mapped_absorption = map_spectrum_to_sequential_scale_full(aligned_absorption, max_val, central_val)
    emission_midi = get_midi_sequence(mapped_emission)
    absorption_midi = get_midi_sequence(mapped_absorption)

    # Generar el gráfico del espectro con una proporción y resolución fijas
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Fijar el tamaño y resolución de la figura
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(full_wavelengths, resultados["full_spectrum"][1], color='gray', label='Flujo Original', alpha=0.5)
    ax.scatter(emission_wl, emission_flux, color='red', s=10, label='Emisión', alpha=0.7)
    ax.scatter(absorption_wl, absorption_flux, color='blue', s=10, label='Absorción', alpha=0.7)
    ax.set_xlabel("Longitud de onda (Å)")
    ax.set_ylabel("Flujo")
    
    if aplicar_rango and rango_valido:
        ax.set_xlim(min_wl, max_wl)
        ax.set_title(f"Espectro con Rango ({min_wl:.1f} - {max_wl:.1f} Å)")
    else:
        ax.set_title("Espectro Original (Emisión y Absorción)")
    
    ax.legend(loc='upper right')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
    
    buf = BytesIO()
    # Guardar la figura con dpi fijo y un pequeño padding para evitar cortes
    fig.savefig(buf, format='png', dpi=150, pad_inches=0.1)
    buf.seek(0)
    combined_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    rango_original = resultados.get('rango_original', (min(wavelength), max(wavelength)))
    rango_actual = (min(full_wavelengths), max(full_wavelengths))
    
    return render_template(
        "preview.html",
        combined_img=combined_img,
        emission_midi_seq=json.dumps(emission_midi),
        absorption_midi_seq=json.dumps(absorption_midi),
        temp_file=file_path,
        nombre_obra=nombre_obra,
        rango_original=json.dumps(rango_original),
        rango_actual=json.dumps(rango_actual),
        es_filtrado=json.dumps(resultados.get('es_filtrado', False))
    )

@app.route('/download', methods=['POST'])
def download():
    file_path = request.form.get('temp_file')
    nombre_obra = request.form.get('nombre_obra', 'MiObra').strip()
    if not file_path or not os.path.exists(file_path):
        return "Archivo no encontrado", 400
    
    wavelength, flux = load_linear_spectrum(file_path)
    resultados = analizar_espectro(wavelength, flux, region_continuo=(5000, 5500), fig=False)
    central_val = resultados["media_continuo"]
    max_val = np.max(resultados["full_spectrum"][1])
    full_wavelengths = resultados["full_spectrum"][0]
    keys = ["emision", "absorcion"]
    
    mapped_spectra = {}
    for k in keys:
        spec_wl = resultados[k][0]
        spec_flux = resultados[k][1]
        aligned = align_spectrum(full_wavelengths, spec_wl, spec_flux, central_val)
        mapped = map_spectrum_to_full_scale(aligned, max_val, central_val)
        mapped_spectra[k] = mapped
        
    temp_files = {}
    for k in keys:
        fd, midi_path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        create_midi_file_otoman_simplified(mapped_spectra[k], midi_path, tempo=488)
        temp_files[k] = midi_path

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.write(temp_files["emision"], arcname=f"{nombre_obra}_emision.mid")
        zip_file.write(temp_files["absorcion"], arcname=f"{nombre_obra}_absorcion.mid")
    
    for path in temp_files.values():
        os.remove(path)
    # Se comenta la eliminación del archivo para mantenerlo disponible:
    # os.remove(file_path)
    
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{nombre_obra}_midi.zip'
    )

@app.route('/library')
def library():
    with open('static/data/sounds.json', 'r', encoding='utf-8') as f:
        sounds_data = json.load(f)
    return render_template("library.html", sounds=sounds_data)

if __name__ == '__main__':
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=puerto)
