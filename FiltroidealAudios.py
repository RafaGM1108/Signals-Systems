import streamlit as st
import soundfile as sf
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from pydub import AudioSegment
import io
import tempfile

# Definir las frecuencias de corte predefinidas
frecuencias_corte = [300, 500, 1000, 5000]

def plot_time_domain(audio_data, sample_rate):
    duration = len(audio_data) / sample_rate
    time = np.linspace(0., duration, len(audio_data))
    
    fig, ax = plt.subplots()
    ax.plot(time, audio_data)
    ax.set(xlabel='Tiempo (s)', ylabel='Amplitud',
           title='Dominio del Tiempo')
    ax.grid()
    
    return fig

def plot_frequency_domain(audio_data, sample_rate):
    n = len(audio_data)
    fft_data = np.fft.fft(audio_data)
    freq = np.fft.fftfreq(n, 1 / sample_rate)
    
    fig, ax = plt.subplots()
    ax.plot(freq, np.abs(fft_data))
    ax.set(xlabel='Frecuencia (Hz)', ylabel='Amplitud',
           title='Dominio de la Frecuencia')
    ax.grid()
    ax.set_xlim(0, 10000)  # Limitar el rango de frecuencias hasta 10 kHz
    
    return fig

def apply_filter(selected_frequency, audio_data, sample_rate):
    n = len(audio_data)
    freq = np.fft.fftfreq(n, 1 / sample_rate)
    mask = np.ones(n)
    selected_frequency1=int(selected_frequency*n/sample_rate)
    mask[selected_frequency1:] = 0
    filtered_audio = np.fft.ifft(np.fft.fft(audio_data) * mask).real
      
    return filtered_audio

def main():
    st.title("Procesamiento de Señal de Audio")

    audio_file = st.file_uploader("Cargar archivo de audio", type=["wav"])

    if audio_file is not None:
        audio_data, sample_rate = sf.read(audio_file)

        st.subheader("Reproductor de Audio Original")
        st.audio(audio_file, format='audio/wav')

        st.subheader("Gráfica en el dominio del tiempo")
        time_fig = plot_time_domain(audio_data, sample_rate)
        st.pyplot(time_fig)

        st.subheader("Gráfica en el dominio de la frecuencia")
        freq_fig = plot_frequency_domain(audio_data, sample_rate)
        st.pyplot(freq_fig)

        st.subheader("Aplicar filtro pasa bajas")
        selected_frequency = st.selectbox("Seleccione una frecuencia de corte:", frecuencias_corte)
        selected_frequency = int(selected_frequency)

        if st.button("Aplicar filtro"):
            filtered_audio = apply_filter(selected_frequency, audio_data, sample_rate)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                sf.write(temp_filename, filtered_audio, sample_rate)

            st.subheader("Reproductor de Audio Filtrado")
            st.audio(temp_filename, format='audio/wav')
            
            st.subheader("Gráfica en el dominio del tiempo (filtrada)")
            filtered_time_fig = plot_time_domain(filtered_audio, sample_rate)
            st.pyplot(filtered_time_fig)

            st.subheader("Gráfica en el dominio de la frecuencia (filtrada)")
            filtered_freq_fig = plot_frequency_domain(filtered_audio, sample_rate)
            st.pyplot(filtered_freq_fig)

if __name__ == "__main__":
    main()
