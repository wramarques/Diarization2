import os
import streamlit as st
import subprocess
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import re
import whisper
#import pysrt
from whisper.utils import get_writer
import whisperTools
import openai
import unicodedata

#############
# Deploy StreamLit
#############
import settings
isStreamlitDeploy=  settings.isStreamlitCloudVersion

# Editialis account
# hugginFace_api_key=os.environ["hugginFace_api_key"]

# ffmepg
# pip install pydub
# pip install   pyannote.audio
# pip install git+https://github.com/openai/whisper.git
# pip install -U webvtt-py
# pysrt

dir_path = os.path.dirname(os.path.realpath(__file__))

def ffmepgExtracWavFile(videoName,dest_file_name):
    # WR Test
    #ffmpeg -i videoName -vn -ac 1 -ar 16000 -ab 320k -y -f wav output.wav
    #videoName="Onliz.m4a"
    #dest_file_name="audio.wav"
    command = "ffmpeg -i "+ dir_path +'/' + videoName + " -vn -ac 1 -ar 16000 -ab 320k -y -f wav "+  dest_file_name
    print(command)
    subprocess.call(command, shell=True)
    #!ffmpeg -i Onliz.m4a -vn -ac 1 -ar 16000 -ab 320k -y -f wav onliz.wav

def _getDiarizationPipeline(use_auth_token,audioFile):
    # Instantiate the pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",use_auth_token=use_auth_token)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"- Device:{device}")    
    if(device=="cuda"):
        pipeline = pipeline.to(torch.device('cuda:0'))

    FILE = {'uri': 'blabla', 'audio': audioFile}

    # Preloading file in memory to speed up treatment
    import torchaudio
    waveform, sample_rate = torchaudio.load(audioFile)
    diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    #  with ProgressHook() as hook:
    #    diarization = pipeline(FILE, hook=hook)
    #diarization = pipeline(FILE)
    return diarization

def writeDiarizationFile(diarizationFileName,dz):
    with open(diarizationFileName, "w") as text_file:
        text_file.write(str(dz))

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

# Editialis account
#openAI_api_key=os.environ["openAI_api_key"]
#openai.api_key = openAI_api_key
#hugginFace_api_key="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" #os.environ["hugginFace_api_key"]

def CreateDir_if_not_exists(directoryName: str) -> bool:
    path = os.path.dirname(os.path.realpath(__file__)) + '/'+ directoryName
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        return True
    return False

dirData=''
xx=CreateDir_if_not_exists(dirData)

dir_path = os.path.dirname(os.path.realpath(__file__))

output_path =  dir_path+'/'+dirData
working_path = dir_path

def normalize_filename(value: str) -> str:
    """
    Normalizes string to ASCII, removes non-alpha characters, and converts spaces to underscores.
    """
    value = str(value)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip()
    return re.sub(r"[-\s]+", "_", value) 

# Disable button after it is clicked
def disable():
    st.session_state.disabled = True

def save_uploadedfile(uploadedfile, fileNameDest ):
    with open(os.path.join(output_path,fileNameDest),"wb") as f:
        f.write(uploadedfile.getbuffer())
    st.success(f"Saved File: {fileNameDest} to working dir.")        

def main():

    #fileToDiarize="Onliz.m4a"
    fileToDiarize="Table2.wav"
    wavFileToDiarize="audio.wav"
    DiarizationTxtFile="diarization.txt"
    DiarizationWaveFile="dz.wav"

    # Check if 'key' already exists in session_state
    # If not, then initialize it
    if 'key' not in st.session_state:
        st.session_state['disabled'] = False
        
    if (not isStreamlitDeploy):
        import ptvsd
        # run application by command streamlit run main.py
        ptvsd.enable_attach(address=('localhost', 8501))
        # Bloque si pas en debug distant
        # ptvsd.wait_for_attach() # Only include this line if you always want to attach the debugger

    st.set_page_config(page_title='Audio Diarization')

    openAI_api_key= st.text_input('openAI Api Key', placeholder='openAI_api_key')
    openai.api_key = openAI_api_key

    hugginFace_api_key= st.text_input('HuggingFace API Key', placeholder='huggin_face_api_key')

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        #bytes_data = uploaded_file.getvalue()
        #st.write( working_path+"/" bytes_data)    
        fileToDiarize=normalize_filename(uploaded_file.name)
        save_uploadedfile(uploaded_file,fileToDiarize)


    if st.button('Run',  on_click=disable, disabled=st.session_state.disabled):
        st.text("Begin of magic !\n - ")

        #vtt_transcribre= whisperTools.whisper_openai_wav_to_vtt(dir_path,"Table2.wav")
        #with open("test.txt", "w") as text_file:
        #text_file.write(str(vtt_transcribre))
        #global_vtt_file =whisperTools.whisper_openai_wav_to_vtt(dir_path,wavFileToDiarize)

        with st.spinner("Working... please wait..."):

            print(f"- fileToDiarize: {fileToDiarize}")
            ffmepgExtracWavFile(fileToDiarize,wavFileToDiarize)

            spacermilli = 2000
            print("- Get Audio from WAV")
            audio = AudioSegment.from_wav(wavFileToDiarize)

            print("- Set spacer")
            spacer = AudioSegment.silent(duration=spacermilli)

            print("- Append spacer")           
            audio = spacer.append(audio, crossfade=0)

            print("- Export wavfile with spacer")           
            audio.export(wavFileToDiarize, format='wav')

            st.text("\t\t- Start Diarzation Pipeline")
            print("- Début diarization pipeline")
            dz=_getDiarizationPipeline(hugginFace_api_key,wavFileToDiarize)
            print("- Fin diarization pipeline")
            st.text("\t\t- End Diarzation Pipeline")

            print(*list(dz.itertracks(yield_label = True))[:10], sep="\n")
            # Ecriture fichier de Diarization (time stamp début/fin et locuteur)
            st.text("\t\t- Write Diarzation file")
            writeDiarizationFile(DiarizationTxtFile,dz)

            # Lecture du fichier de Diarization et création de la liste de dzList qui qui comprend : tiestampStart,timeStampEnd, isLex (speaker 01)
            dz = open(DiarizationTxtFile).read().splitlines()
            dzList = []
            for l in dz:
                start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
                start = millisec(start) - spacermilli
                end = millisec(end)  - spacermilli
                lex = not re.findall('SPEAKER_01', string=l)
                speaker=re.findall('SPEAKER_\d+', string=l)[0]
                dzList.append([start, end, lex,speaker])

            # Debug de controle
            print(*dzList[:10], sep='\n')

            # Preparing audio file from the diarization - Add spacer after each segment and produce
            # a specific wav file with spacer added.
            # sounds is an audio segment
            st.text("\t\t-Prepare audio file for Diarirzation")

            sounds = spacer
            segments = []

            dz = open(DiarizationTxtFile).read().splitlines()
            for l in dz:
                start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
                start = int(millisec(start)) #milliseconds
                end = int(millisec(end))  #milliseconds

                # Ajout de la longeur (en ms) du segment total après chaque itération(première itération un spacer 2000ms par défaut)
                # [2000, 6649, 9464, 16812, 19372, 26686, 29586, 34302]
                # 
                segments.append(len(sounds))

                # Ajout dans sound du segment correspondant (audio est le wav fichier à diarizer)
                sounds = sounds.append(audio[start:end], crossfade=0)
                #Ajout du spacer
                sounds = sounds.append(spacer, crossfade=0)

            # Sauvegarde du wav avec les segments de spacer
            sounds.export(DiarizationWaveFile, format="wav") #Exports to a wav file in the current path.

            # Debug, affichage des premiers segments
            print(segments[:8])

            #command = "whisper dz.wav --language fr --model large --output_format vtt --output_dir " +dir_path
            #print(command)      
            #subprocess.call(command, shell=True)

            # Application du modèle Whisper sur le wav de diarizaiton (avec les spacers)
            print("Début application du model Whisper")
            st.text("\t\t-Generate global vtt file")
            # base small medium large
            #model = whisper.load_model("small")
            #audio=DiarizationWaveFile
            #result = model.transcribe(audio, language='fr')
            global_vtt_file =whisperTools.whisper_openai_wav_to_vtt(dir_path,wavFileToDiarize)
            #print("Fin d'application du model Whisper")

            # Sauvegarde au format VTT
            # vtt is similar to the SRT format except that it accommodates text formatting, positioning, and rendering options 
            # (pop-up, roll-on, paint-on, etc.). It has quickly gained popularity because it is the caption format of choice for HTML5 text track rendering
            # Save as a VTT file
            # Set some initial options values
            #options = {
            #    'max_line_width': None,
            #    'max_line_count': None,
            #    'highlight_words': False
            #}
            #vtt_writer = get_writer("vtt", dir_path)
            #vtt_writer(result, audio,options)

            # Lecture du fichier VTT dans la variable caption
            import webvtt
            captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(global_vtt_file)]
            print(*captions[:8], sep='\n')

            preS = '''
                    <!DOCTYPE html>\n<html lang="fr">\n  
                    <head>\n    
                        <meta charset="UTF-8">\n    
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">\n    
                        <meta http-equiv="X-UA-Compatible" content="ie=edge">\n    
                        <title>Lexicap</title>\n    
                        <style>\n        
                            body {\n            font-family: sans-serif;\n            font-size: 18px;\n            color: #111;\n            padding: 0 0 1em 0;\n        }\n
                            .l {\n          color: #050;\n        }\n        
                            .s {\n            display: inline-block;\n        }\n        
                            .e {\n            display: inline-block;\n        }\n        
                            .t {\n            display: block; clear:both; margin-top:30px;\n        }\n        
                        </style>\n  
                    </head>\n  
                    <body>\n    <h2>Video Diarization</h2>\n  
                    <br>\n'''
            postS = '\t</body>\n</html>'

            from datetime import timedelta

            html = list(preS)

            speaker=dzList[0][3]
            activateSpeakerLabel=True
            lexicap=""

            for i in range(len(segments)):
                idx = 0
                for idx in range(len(captions)):
                    if captions[idx][0] >= (segments[i] - spacermilli):
                        break;

                while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
                    c = captions[idx]

                    print(f"{str(c[0])}-{str(c[1])}:{c[2]}")
                    
                    start = dzList[i][0] + (c[0] -segments[i])
                    if(speaker!=dzList[i][3] or idx==0):
                        speaker=dzList[i][3]
                        activateSpeakerLabel=True
                    else:
                        activateSpeakerLabel=False

                    if start < 0:
                        start = 0
                    idx += 1

                    start = start / 1000.0
                    startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600),
                                                            (int)(start % 3600 // 60),
                                                            start % 60)
                    if(activateSpeakerLabel):
                        html.append(f'\t\t\t\t<div class="t"><strong>{speaker}:</strong> </div>\n')
                        lexicap=lexicap+f"\n\n**{speaker}**\n"

                    #html.append('\t\t\t<div class="c">\n')
                    #html.append(f'\t\t\t\t<a class="l" href="#{startStr}" id="{startStr}">link</a> |\n')
                    #html.append(f'\t\t\t\t<div class="s"><a href="javascript:void(0);" onclick=setCurrentTime({int(start)})>{startStr}</a></div>\n')
                    #html.append(f'\t\t\t\t<div class="t">{"[Lex]" if dzList[i][2] else "[Yann]"} {c[2]}</div>\n')
                    #html.append(f'\t\t\t\t<div class="t">{c[2]}</div>\n')
                    html.append(f'{c[2]}')
                    lexicap=lexicap+f"{c[2]} "
                    #html.append('\t\t\t</div>\n\n')

            html.append(postS)
            s = "".join(html)

            with open(file="lexicap.html", mode="w",encoding="utf8") as text_file:
                text_file.write(s)
                text_file.write(lexicap)
            #st.text_area("Lexicap", s,height=400)
            #print(s)
            st.markdown(lexicap, unsafe_allow_html=True)

if __name__ == '__main__':
    main()