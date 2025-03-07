# microtonal.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from astropy.io import fits
import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
from midiutil import MIDIFile

#############################################
# Sección 1: Funciones para Escala Microtonal
#############################################

def create_microtonal_scale_otoman(octaves=(1, 7)):
    base_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    steps_per_octave = 53
    steps_per_base = [5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4]
    scale = []
    
    for octave in range(octaves[0], octaves[1] + 1):
        for i, base_note in enumerate(base_notes):
            num_steps = steps_per_base[i]
            for step in range(num_steps):
                if num_steps == 5:
                    if step == 0:
                        note = f"{base_note}{octave}"
                    elif step == 1:
                        note = f"^{base_note}{octave}"
                    elif step == 2:
                        note = f"^^{base_note}{octave}"
                    elif step == 3:
                        next_base = base_notes[(i + 1) % len(base_notes)]
                        note = f"vv{next_base}{octave}"
                    elif step == 4:
                        next_base = base_notes[(i + 1) % len(base_notes)]
                        note = f"v{next_base}{octave}"
                elif num_steps == 4:
                    if step == 0:
                        note = f"{base_note}{octave}"
                    elif step == 1:
                        note = f"^{base_note}{octave}"
                    elif step == 2:
                        note = f"^^{base_note}{octave}"
                    elif step == 3:
                        next_base = base_notes[(i + 1) % len(base_notes)]
                        note = f"vv{next_base}{octave}"
                scale.append(note)
    
    center_index = len(scale) // 2
    scale.insert(center_index, 'silence')
    return scale

def display_octaves_otoman_simplified(scale, octaves=(1, 7)):
    steps_per_octave = 53
    for octave in range(octaves[0], octaves[1] + 1):
        start = (octave - octaves[0]) * steps_per_octave
        end = start + steps_per_octave
        octave_notes = scale[start:end]
        df = pd.DataFrame({'Nota': octave_notes})
        print(f"\n#### Octava {octave}\n")
        print(df.to_markdown(index=False))

#############################################
# Sección 2: Funciones para Conversión a MIDI
#############################################

def split_note_octave_microtonal_otoman_simplified(note):
    if note == 'silence':
        return None
    match = re.match(r"([\^v]*)([A-G]#?)(\d+)", note)
    if match:
        modifiers = match.group(1)
        note_base = match.group(2)
        octave = int(match.group(3))
        return (modifiers, note_base, octave)
    return None

def note_to_midi_microtonal_otoman_simplified(modifiers, note_base, octave, divisions_per_octave=53):
    note_dict = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
                 'E': 4, 'F': 5, 'F#': 6, 'G': 7,
                 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    midi_number = note_dict.get(note_base, 0) + (octave + 1) * 12

    microtone_steps = 0
    if modifiers == '^^':
        microtone_steps = 2
    elif modifiers == '^':
        microtone_steps = 1
    elif modifiers == 'vv':
        microtone_steps = -2
    elif modifiers == 'v':
        microtone_steps = -1

    cents = microtone_steps * (1200 / divisions_per_octave)
    pitch_bend = int((cents / 200) * 8192)
    pitch_bend = max(-8192, min(8191, pitch_bend))
    return midi_number, pitch_bend

def create_midi_file_otoman_simplified(mapped_spectrum, file_name, tempo=120, note_duration=1, divisions_per_octave=53):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    microseconds_per_beat = mido.bpm2tempo(tempo)
    track.append(MetaMessage('set_tempo', tempo=microseconds_per_beat))
    
    ticks_per_beat = mid.ticks_per_beat
    ticks_per_note = int(note_duration * ticks_per_beat)
    
    delta_time = 0
    for note in mapped_spectrum:
        messages = []
        if note != 'silence':
            note_info = split_note_octave_microtonal_otoman_simplified(note)
            if note_info:
                modifiers, note_base, octave = note_info
                midi_num, pitch_bend = note_to_midi_microtonal_otoman_simplified(modifiers, note_base, octave, divisions_per_octave)
                if pitch_bend != 0:
                    messages.append(Message('pitchwheel', pitch=pitch_bend, time=delta_time))
                    delta_time = 0
                    messages.append(Message('note_on', note=midi_num, velocity=100, time=0))
                else:
                    messages.append(Message('note_on', note=midi_num, velocity=100, time=delta_time))
                    delta_time = 0
                messages.append(Message('note_off', note=midi_num, velocity=0, time=ticks_per_note))
                for msg in messages:
                    track.append(msg)
            else:
                messages.append(Message('note_off', note=0, velocity=0, time=ticks_per_note))
                delta_time = 0
                for msg in messages:
                    track.append(msg)
        else:
            messages.append(Message('note_off', note=0, velocity=0, time=ticks_per_note))
            delta_time = 0
            for msg in messages:
                track.append(msg)
    mid.save(file_name)

#############################################
# Sección 3: Funciones para Mapeo del Espectro
#############################################

def map_spectrum_to_full_scale(spectrum, max_val, central_val):
    full_scale = create_microtonal_scale_otoman()
    center_index = full_scale.index('silence')
    num_notes_up = len(full_scale) - center_index - 1
    num_notes_down = center_index
    
    step = max_val / max(num_notes_up, num_notes_down)
    upper_silence_limit = central_val + step / 2
    lower_silence_limit = central_val - step / 2
    
    spectrum = np.array(spectrum)
    mapped_spectrum = np.full(spectrum.shape, 'silence', dtype=object)
    
    above_silence = spectrum > upper_silence_limit
    displacement_up = np.floor((spectrum[above_silence] - central_val) / step).astype(int)
    displacement_up = np.clip(displacement_up, 0, num_notes_up - 1)
    mapped_spectrum[above_silence] = [full_scale[center_index + 1 + d] for d in displacement_up]
    
    below_silence = spectrum < lower_silence_limit
    displacement_down = np.floor((central_val - spectrum[below_silence]) / step).astype(int)
    displacement_down = np.clip(displacement_down, 0, num_notes_down - 1)
    mapped_spectrum[below_silence] = [full_scale[center_index - 1 - d] for d in displacement_down]
    
    return mapped_spectrum.tolist()

def map_spectrum_to_sequential_scale_full(spectrum, max_val, central_map):
    full_scale = create_microtonal_scale_otoman()
    note_to_sequential_dict = {note: i + 1 for i, note in enumerate(full_scale)}
    mapped_spectrum = map_spectrum_to_full_scale(spectrum, max_val, central_map)
    mapped_array = np.array(mapped_spectrum)
    vectorized_get = np.vectorize(note_to_sequential_dict.get)
    sequential_mapped = vectorized_get(mapped_array)
    return sequential_mapped.tolist(), mapped_spectrum

#############################################
# Sección 4: Funciones de Análisis del Espectro
#############################################

def load_linear_spectrum(file_path):
    # Desactivar el memory mapping para evitar bloqueos en Windows
    with fits.open(file_path, memmap=False) as hdulist:
        header = hdulist[0].header
        flux = hdulist[0].data
        crval1 = header['CRVAL1']
        crpix1 = header['CRPIX1']
        cdelt1 = header['CDELT1']
        num_pixels = header['NAXIS1']
        wavelength = crval1 + (np.arange(num_pixels) - (crpix1 - 1)) * cdelt1
    return wavelength, flux

def analizar_espectro(wavelength, flux, region_continuo, fig=False):
    mask_continuo = (wavelength > region_continuo[0]) & (wavelength < region_continuo[1])
    continuo_flux = flux[mask_continuo]
    
    media_continuo = np.mean(continuo_flux)
    desviacion_continuo = np.std(continuo_flux)
    
    mask_emision = flux >= media_continuo
    mask_absorcion = flux < media_continuo
    
    espectro_emision = (wavelength[mask_emision], flux[mask_emision])
    espectro_absorcion = (wavelength[mask_absorcion], flux[mask_absorcion])
    
    if fig:
        plt.figure(figsize=(12, 6))
        plt.plot(wavelength, flux, label='Espectro original', color='gray', alpha=0.6)
        plt.scatter(espectro_emision[0], espectro_emision[1], label='Emisión', color='red', s=10)
        plt.scatter(espectro_absorcion[0], espectro_absorcion[1], label='Absorción', color='blue', s=10)
        plt.fill_between(wavelength, media_continuo - desviacion_continuo, media_continuo + desviacion_continuo,
                         color='gray', alpha=0.3, label=f'Ruido (1σ = {desviacion_continuo:.2f})')
        plt.axhline(media_continuo, color='black', linestyle='--', label=f'Media del continuo ({media_continuo:.2f})')
        plt.xlabel('Longitud de onda (Å)')
        plt.ylabel('Flujo Normalizado')
        plt.title('División del Espectro en Emisión y Absorción')
        plt.legend()
        plt.grid()
        plt.show()
        
    return {
        'emision': espectro_emision,
        'absorcion': espectro_absorcion,
        'media_continuo': media_continuo,
        'desviacion_continuo': desviacion_continuo,
        'full_spectrum': (wavelength, flux)
    }

def align_spectrum(full_wavelengths, spectrum_wavelengths, spectrum_values, central_val):
    wavelength_to_value = dict(zip(spectrum_wavelengths, spectrum_values))
    aligned_spectrum = [wavelength_to_value.get(wl, central_val) for wl in full_wavelengths]
    return aligned_spectrum

if __name__ == "__main__":
    scale = create_microtonal_scale_otoman(octaves=(1, 7))
    display_octaves_otoman_simplified(scale, octaves=(1, 7))
