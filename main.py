# ##################################################
# Project Diarization based on:
#       - Pyanote for Diarization process
#       - Streamlit for user interface
#       - Google Cloud to store steps in process flow
#       - User access control defined in config.yaml => a folder by user where datas are stored (see below how to generate password)
# ##################################################
# Google cloud elements (see gcp_service_account section in secret.toml file )
# ##################################################
# IAM Access https://console.cloud.google.com/iam-admin/serviceaccounts/details/103394789504793451452/keys?hl=fr&project=diarization-407413
# Google Cloud BigQuery Database => https://console.cloud.google.com/bigquery?hl=fr
#                                   https://console.cloud.google.com/bigquery?hl=fr&project=diarization-407413&ws=!1m0
# https://console.cloud.google.com/bigquery?hl=fr&project=diarization-407413&ws=!1m10!1m4!4m3!1sdiarization-407413!2sdiarization!3sProcess!1m4!1m3!1sdiarization-407413!2sbquxjob_52202be3_18c44b90635!3sEU

import os
import streamlit as st
import subprocess
from pydub import AudioSegment
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database import loader
from pyannote.database import util
# Util files to construct segments
from pyannote_whisper.utils import dz_segment_to_text

#import pysrt
from whisper.utils import get_writer
import whisperTools
import fileTools
import openai
from PIL import Image
import pandas as pd

#############
# Use this method to geneate a password to store in yaml file
#############
#auth = sa.Authenticator(
#    "fdDGdffgf45645656465",
#   token_url="/token",
#    token_ttl=3600,
#   password_hashing_method=sa.PasswordHashingMethod.BCRYPT,
#)
#hashed_passwords = sa.Hasher(['password de will']).generate()

#############
# Manage Authentification
# https://blog.streamlit.io/streamlit-authenticator-part-1-adding-an-authentication-component-to-your-app/
#############
import streamlit_authenticator as sa
import yaml
from yaml.loader import SafeLoader

#############
# Deploy StreamLit
#############
import settings
isStreamlitDeploy=  settings.isStreamlitCloudVersion

########
#Activate login
########
activateLogin=True
if(not isStreamlitDeploy):
    activateLogin=True

# Editialis account
# hugginFace_api_key=os.environ["hugginFace_api_key"]

# ffmepg
# pip install pydub
# pip install   pyannote.audio
# pip install git+https://github.com/openai/whisper.git
# pip install -U webvtt-py
# pysrt

dir_path = os.path.dirname(os.path.realpath(__file__))

cwd = os.getcwd()
with open(cwd+'/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = sa.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

def ffmepgExtracWavFile(dir_path,videoName,dest_file_name, limitSeconds=-1):
    # WR Test
    #ffmpeg -i videoName -vn -ac 1 -ar 16000 -ab 320k -y -f wav output.wav
    #videoName="Onliz.m4a"
    #dest_file_name="audio.wav"
    # t time in seconds
    if(limitSeconds!=-1):
        command = "ffmpeg -i "+ dir_path +'/' + videoName + "  -t "+ str(limitSeconds) +" -vn -ac 1 -ar 16000 -ab 320k -y -f wav "+ dir_path + '/' + dest_file_name
    else:
        command = "ffmpeg -i "+ dir_path +'/' + videoName + " -vn -ac 1 -ar 16000 -ab 320k -y -f wav "+ dir_path + '/' + dest_file_name
    print(command)
    subprocess.call(command, shell=True)
    #!ffmpeg -i Onliz.m4a -vn -ac 1 -ar 16000 -ab 320k -y -f wav onliz.wav

def _getDiarizationPipeline(use_auth_token,dir_path,audioFile, activatePreloadMemory=False):
    # Instantiate the pipeline
    print (f" Token size : {len(use_auth_token)} ")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=use_auth_token)
    #pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",use_auth_token=use_auth_token)

    import torch
    # cuda if gpu detected
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"- Device:{device}")    

    print(f"- File: {dir_path +'/'+ audioFile}")    

    if(device=="cuda"):
        pipeline = pipeline.to(torch.device('cuda:0'))

    FILE = {'uri': 'blabla', 'audio': dir_path +'/'+ audioFile}

    diarization= None
    if(activatePreloadMemory):
        print(f"- Preloading file in memory to speed up treatment")    
        # Preloading file in memory to speed up treatment
        import torchaudio
        waveform, sample_rate = torchaudio.load(dir_path+'/'+audioFile)
        # return  Segments : Tuple with additionnal features 
        # https://pyannote.github.io/pyannote-core/structure.html#segment
        # https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb

        # Le hook pose pb en prod
        if isStreamlitDeploy:
            diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        else:
            with ProgressHook() as hook:
                diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    else:
        if isStreamlitDeploy:
            diarization = pipeline(dir_path+'/'+audioFile)
        else:
            with ProgressHook() as hook:
                diarization = pipeline(dir_path+'/'+audioFile, hook=hook)
    print ("End diarization process")
    return diarization

def writeDiarizationFile(diarizationFileName,dz):
    with open(diarizationFileName, "w") as text_file:
        text_file.write(str(dz))

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def disable():
    """
    Disable button after it is clicked
    """
    st.session_state.disabled = True

def main_application(dir_path,output_path,working_path):
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

    if (isStreamlitDeploy and not activateLogin):
            openAI_api_key= st.text_input('openAI Api Key', placeholder='openAI_api_key')
            hugginFace_api_key= st.text_input('HuggingFace API Key', placeholder='huggin_face_api_key')
    else:
        # En déploiement sur HF les clés sont récupérées via le paramétrage dans les settings
        if (isStreamlitDeploy):
            openAI_api_key= os.environ["openAI_api_key"]
            hugginFace_api_key= os.environ["hugginFace_api_key"]            
        else:
            # En local on passe par le fichier toml
            openAI_api_key= st.secrets.openai.openAI_api_key
            hugginFace_api_key= st.secrets.huggingface.hugginFace_api_key

    openai.api_key = openAI_api_key
    # Set default variable
    optionDiarizationFile=""

    radioDiarizationWaveToMemory = st.radio(    "Diarization : Load wave in memory  ", ('No', 'Yes'))
    radioActivateWhisperAPI  = st.radio(    "Whisper via API ", ('Yes', 'No'))

    activateDiarizationWaveToMemory = True if radioDiarizationWaveToMemory=="Yes" else False    
    activateActivateWhisperAPI = True if radioActivateWhisperAPI=="Yes" else False    

    # Get Wav Files 
    dfWav_Files= fileTools.directory_to_dataframe(working_path,".wav")
    if( dfWav_Files is not None and len(dfWav_Files)>0):
        listWaveFiles=list(dfWav_Files['Name'])
        listWaveFiles.insert(0,'-')
        optionWaveFile = st.selectbox('Wave files to process',listWaveFiles )

    # Get DZ Files in rttm format located in user folder 
    dfDiarzation_Files= fileTools.directory_to_dataframe(working_path,".rttm")
    if(dfDiarzation_Files is not None and len(dfDiarzation_Files)>0):
        listDiarzation=list(dfDiarzation_Files['Name'])
        listDiarzation.insert(0,'-')
        optionDiarizationFile = st.selectbox('Diarization file to process',listDiarzation )

    # Get ASR Files located in user folder 
    dfASR_Files= fileTools.directory_to_dataframe(working_path,".asr_result")
    if(dfASR_Files is not None and len(dfASR_Files)>0):
        listASR=list(dfASR_Files['Name'])
        listASR.insert(0,'-')
        optionASRFile = st.selectbox('ASR File to process',listASR )
        #st.write('You selected:', optionASRFile)

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # Get normalized name of file of uploaded file (video, audio..)
        fileToDiarize=fileTools.normalize_filename(uploaded_file.name)
        fileTools.save_uploadedfile(working_path,uploaded_file,fileToDiarize)

    if st.button('Run',  on_click=disable, disabled=st.session_state.disabled):
        st.text("Begin of magic !\n")

        with st.spinner("Working... please wait..."):

            if ('-' in optionDiarizationFile and '-' in optionASRFile and '-' in optionWaveFile):

                print(f"- fileToDiarize: {fileToDiarize}")

                if isStreamlitDeploy:
                    limitSecondsVideoToProcess=-1
                else:
                    limitSecondsVideoToProcess=-1

                # Add Prefixe to extract wav file that will be pass to Diarization process
                wavFileToDiarize = "dz-"+ fileTools.forceFileNameExtension(fileToDiarize, "wav")
                ffmepgExtracWavFile(output_path, fileToDiarize,wavFileToDiarize, limitSeconds=limitSecondsVideoToProcess)

                # Add spacer to file
                spacermilli = 5000
                print("- Get Audio from WAV")
                audio = AudioSegment.from_wav(output_path+"/"+wavFileToDiarize)

                print("- Define spacer")
                spacer = AudioSegment.silent(duration=spacermilli)

                print("- Add spacer on begging of wave file ")           
                audio = spacer.append(audio, crossfade=0)

                print(f"- Export wavfile with spacer added {wavFileToDiarize}] ")           
                audio.export(output_path + "/" +wavFileToDiarize, format='wav')
                # / Add spacer to file

                st.audio(output_path+"/"+wavFileToDiarize, format="audio/wav", start_time=0)

                st.text("\t\t- Start Diarization Pipeline")
                print("- Début diarization pipeline")
                dz=_getDiarizationPipeline(hugginFace_api_key, output_path, wavFileToDiarize, activateDiarizationWaveToMemory)
                # Backup dz file for debug usage
                #fileTools.writeJsonFile(output_path + "/"+ fileTools.forceFileNameExtension(fileToDiarize, "dz"), dz._tracks)
                print("- End diarization pipeline")
                st.success(' Diarization Pipeline', icon="✅")

                # For debugging purpose- Save diarization file
                dzFileContent="\n".join(dz_segment_to_text(dz))
                fileTools.writeTextFile(output_path + "/"+ fileTools.forceFileNameExtension(fileToDiarize, "dz"), dzFileContent)

                # dump the diarization output to disk using RTTM format
                with open(output_path + "/"+ fileTools.forceFileNameExtension(fileToDiarize, "rttm"),'w') as rttm:
                    dz.write_rttm(rttm)                    
            else:
                #dz = loader.RTTMLoader(output_path + "/"+ optionDiarizationFile)
                loadRTTM= util.load_rttm(output_path + "/"+ optionDiarizationFile,  keep_type="SPEAKER")
                firstKey=list(loadRTTM.keys())[0]
                dz=loadRTTM[firstKey]
                wavFileToDiarize=optionWaveFile

            final_diarization=whisperTools.diarize_wav(output_path,wavFileToDiarize,dz,whisperViaApi=activateActivateWhisperAPI)

            currentSpk= ""
            backupTxt=""
            segmentTxt=""
            for seg, spk, sent in final_diarization:
                line=""
                segmentTxt = segmentTxt + f'{seg.start:.2f} => {seg.end:.2f} {spk}\n '
                if(currentSpk!=spk and spk is not None):
                    line = line + f'## {spk}\n'
                    currentSpk=spk

                line = line + f'{sent}'
                print(line)
                st.markdown(line, unsafe_allow_html=True)
                backupTxt = backupTxt +  line +'\n'

            debugSegments =  '\n\n\n'+ "Debug segments" +'\n' + segmentTxt

            with st.expander("Debug"):
                splitDebug = debugSegments.split("\n")
                for txt in splitDebug:
                    st.write(txt)
            
            backupTxt = backupTxt + debugSegments

            fileTools.writeTextFile( output_path+"/"+ wavFileToDiarize+".transcript.txt" ,backupTxt)

def setPath():

    dirData=''

    dir_path = os.path.dirname(os.path.realpath(__file__))

    if(st.session_state["authentication_status"]):
        resultCreate= fileTools.CreateDir_if_not_exists(st.session_state["username"] )
        output_path =  dir_path+'/'+ st.session_state["username"] 
        working_path = output_path 
    else:
        resultCreate= fileTools.CreateDir_if_not_exists(dirData)
        output_path =  dir_path
        working_path = output_path 

    return dir_path,output_path,working_path

def main():
    #st.set_page_config(page_title='Audio Diarization')

    image = Image.open('logo.png')
    st.image(image,width=400 )


#    if (not isStreamlitDeploy):
        #dirData='data'
#        dirData=''

    if activateLogin:
        name, authentication_status, username = authenticator.login('Login', 'main')

        if authentication_status:
            authenticator.logout('Logout', 'main')
            dir_path,output_path,working_path =setPath()

            # Get Files located in user folder 
            dfFiles= fileTools.directory_to_dataframe(working_path)

            # Write list files in  
            if(not (dfFiles is None) and len(dfFiles)>0):
                fileTools.create_tabe_df(dfFiles,st.sidebar)
            
            main_application(dir_path,output_path,working_path)
            #st.write(f'Welcome *{name}*')
            #st.title('Some content')
        elif authentication_status == False:
            st.error('Username/password is incorrect')
        elif authentication_status == None:
            st.warning('Please enter your username and password')
    else:
        dir_path,output_path,working_path =setPath()
        main_application(dir_path,output_path,working_path)

if __name__ == '__main__':
    main()