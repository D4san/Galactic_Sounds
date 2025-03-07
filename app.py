# app.py

from flask import Flask, request, render_template, send_file, redirect, url_for
import numpy as np
import os
import tempfile
import zipfile
from io import BytesIO
import json
import base64

from microtonal import (
    load_linear_spectrum,
    analizar_espectro,
    align_spectrum,
    map_spectrum_to_full_scale,
    map_spectrum_to_sequential_scale_full,
    create_midi_file_otoman_simplified,
    split_note_octave_microtonal_otoman_simplified,
    note_to_midi_microtonal_otoman_simplified
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
    return render_template('index.html')

@app.route('/preview', methods=['GET', 'POST'])
def preview():
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    archivo = request.files['archivo']
    nombre_obra = request.form.get('nombre', 'MiObra').strip()
    if not archivo:
        return "No se recibió archivo", 400

    # Guardar el archivo FITS temporalmente
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".fits")
    archivo.save(temp.name)
    temp_path = temp.name
    temp.close()

    # Procesar el espectro
    wavelength, flux = load_linear_spectrum(temp_path)
    resultados = analizar_espectro(wavelength, flux, region_continuo=(5000, 5500), fig=False)
    
    # Extraer datos
    emission_wl, emission_flux = resultados["emision"]
    absorption_wl, absorption_flux = resultados["absorcion"]
    central_val = resultados["media_continuo"]
    max_val = np.max(resultados["full_spectrum"][1])
    full_wavelengths = resultados["full_spectrum"][0]
    
    # Alinear emisión y absorción
    aligned_emission = align_spectrum(full_wavelengths, emission_wl, emission_flux, central_val)
    aligned_absorption = align_spectrum(full_wavelengths, absorption_wl, absorption_flux, central_val)
    
    # Mapear a notas y generar secuencias MIDI
    _, mapped_emission = map_spectrum_to_sequential_scale_full(aligned_emission, max_val, central_val)
    _, mapped_absorption = map_spectrum_to_sequential_scale_full(aligned_absorption, max_val, central_val)
    emission_midi = get_midi_sequence(mapped_emission)
    absorption_midi = get_midi_sequence(mapped_absorption)

    # Generar el gráfico (solo flujo)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(full_wavelengths, resultados["full_spectrum"][1], color='gray', label='Flujo Original', alpha=0.5)
    ax.scatter(emission_wl, emission_flux, color='red', s=10, label='Emisión', alpha=0.7)
    ax.scatter(absorption_wl, absorption_flux, color='blue', s=10, label='Absorción', alpha=0.7)
    ax.set_xlabel("Longitud de onda (Å)")
    ax.set_ylabel("Flujo")
    ax.set_title("Espectro Original (Emisión y Absorción)")
    ax.legend(loc='upper right')
    
    # Ajustar márgenes para reducir espacio en blanco
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    
    buf = BytesIO()
    # Usar bbox_inches='tight' y pad_inches=0 para recortar aún más
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    combined_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return render_template(
        "preview.html",
        combined_img=combined_img,
        emission_midi_seq=json.dumps(emission_midi),
        absorption_midi_seq=json.dumps(absorption_midi),
        temp_file=temp_path,
        nombre_obra=nombre_obra
    )

@app.route('/download', methods=['POST'])
def download():
    temp_file = request.form.get('temp_file')
    nombre_obra = request.form.get('nombre_obra', 'MiObra').strip()
    if not temp_file or not os.path.exists(temp_file):
        return "Archivo no encontrado", 400
    
    # Volver a procesar el espectro y generar archivos MIDI
    wavelength, flux = load_linear_spectrum(temp_file)
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
        
    # Crear archivos MIDI temporales
    temp_files = {}
    for k in keys:
        fd, midi_path = tempfile.mkstemp(suffix=".mid")
        os.close(fd)
        create_midi_file_otoman_simplified(mapped_spectra[k], midi_path, tempo=488)
        temp_files[k] = midi_path

    # Empaquetar en ZIP
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        zip_file.write(temp_files["emision"], arcname=f"{nombre_obra}_emision.mid")
        zip_file.write(temp_files["absorcion"], arcname=f"{nombre_obra}_absorcion.mid")
    
    # Eliminar archivos MIDI temporales y el FITS
    for path in temp_files.values():
        os.remove(path)
    os.remove(temp_file)
    
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'{nombre_obra}_midi.zip'
    )

import os

if __name__ == '__main__':
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=puerto)

