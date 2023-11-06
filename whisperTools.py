import os
import math
from pydub import AudioSegment
import streamlit as st
import openai
import re
from whisper.utils import get_writer
from webvtt import WebVTT, Caption

def chunk_wav_file(output_path,wav_file_name, chunk_threshold_mb=25):

    file_size = os.path.getsize(output_path +"/"+ wav_file_name)
    file_size_mb=int(file_size)/1024/1024

    if(file_size_mb>chunk_threshold_mb):
        chunks_total=math.ceil(file_size_mb/chunk_threshold_mb)
    else:
        wav_file = AudioSegment.from_wav(output_path +"/"+ wav_file_name)
        wav_file.export(output_path+"/0-"+ wav_file_name , format="wav")
        return 1
    
    # pip insall AudioSegment
    # need => pip install ffprobe 
    wav_file = AudioSegment.from_wav(output_path +"/"+ wav_file_name)

    chunk_seconds_step =math.ceil(wav_file.duration_seconds/chunks_total)
    ## PyDub handles time in milliseconds
    #ten_minutes = 10 * 60 * 1000
    for x in range(chunks_total):
        start = x*chunk_seconds_step*1000
        end = (x+1)*chunk_seconds_step*1000
        
        wav_chunk = wav_file[start:end]
        wav_chunk.export(output_path+"/"+ str(x) +"-"+ wav_file_name , format="wav")
        print(f" creating chunk wav {str(x)}-{wav_file_name}")
    return  chunks_total

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def millisec_to_hms(millisec):
    seconds, milliseconds = divmod(millisec,1000) 
    minutes, seconds = divmod(seconds, 60) 
    hours, minutes = divmod(minutes, 60) 
    days, hours = divmod(hours, 24) 
    #seconds = seconds + milliseconds/1000 

    strHour=""
    strMinutes=""
    strSeconds=""
    strMilliSec=""

    if(hours>=10):
        strHour=str(hours)
    else:
        strHour="0"+str(hours)     

    if(minutes>=10):
        strMinutes=str(minutes)
    else:
        strMinutes="0"+str(minutes)     

    if(seconds>=10):
        strSeconds=str(seconds)
    else:
        strSeconds="0"+str(seconds)     

    if (milliseconds<10):
        strMilliSec="00"+ str(milliseconds)
    elif(milliseconds<100 and milliseconds>=10 ):
        strMilliSec="0"+ str(milliseconds)
    else:
        strMilliSec=str(milliseconds)        
    return f"{strHour}:{strMinutes}:{strSeconds}.{strMilliSec}" 

def whisper_openai_wav_to_vtt(output_path,wav_file_name):

    # Whisper is an Automatic Speech Recognition (ASR) developped by OpenIA
    # https://github.com/openai/whisper
    result = ""
    global_vtt_file="global_"+wav_file_name.replace(".wav",".vtt")
    
    chunks_total =chunk_wav_file(output_path,wav_file_name)
    
    # load entire file
    # Auto Chuncking with opena ai audio transcribe
    # https://platform.openai.com/docs/guides/speech-to-text
    #my_bar = st.progress(1, text="Processing chunks")
    if True:
        for x in range(chunks_total):

            #my_bar.progress((x+1)/chunks_total, text=" Processing chunks: " + str(x+1) + "/"+str(chunks_total))
            file_to_process= str(x) + "-" + wav_file_name
            fullPath_file_to_process=output_path+"/" + file_to_process
            print(f" processing openAI transcribe for : {fullPath_file_to_process}")
            audio_file = open(fullPath_file_to_process, "rb")

            # https://platform.openai.com/docs/guides/speech-to-text
            # The Whisper v2-large model is currently available through our API with the whisper-1 model name.
            # Underlying api call : https://platform.openai.com/docs/api-reference/audio/create
            # See prompt parameter
            # https://platform.openai.com/docs/guides/speech-to-text
            result = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio_file,
                language="fr",
                response_format="vtt"
            )

            vtt_file_name=file_to_process.replace(".wav",".vtt")
            # Save as a VTT file
            with open(file=vtt_file_name, mode="w",encoding="utf8") as vtt_file:
                vtt_file.write(result)

    offsetMilliSeconds=0
    vtt = WebVTT()

    for x in range(chunks_total):
        file_to_process= str(x) + "-" + wav_file_name
        vtt_file_name=file_to_process.replace(".wav",".vtt")

        # Lecture du fichier VTT dans la variable caption
        captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in WebVTT.read(vtt_file_name)]
        print(*captions[:8], sep='\n')

        # Get offset by last elements in captions for first file
        if(x==0):
            offsetMilliSeconds=offsetMilliSeconds+captions[len(captions)-1][1]
            for idxCaption in range(len(captions)-1):

                # creating global caption
                caption = Caption(
                    millisec_to_hms(captions[idxCaption][0]),
                    millisec_to_hms(captions[idxCaption][1]),
                    captions[idxCaption][2]
                )
                vtt.captions.append(caption)

        # Apply offset
        if(x>0):
            for idxCaption in range(len(captions)-1):
                captions[idxCaption][0]=captions[idxCaption][0]+offsetMilliSeconds
                captions[idxCaption][1]=captions[idxCaption][1]+offsetMilliSeconds

                # creating global caption
                caption = Caption(
                    millisec_to_hms(captions[idxCaption][0]),
                    millisec_to_hms(captions[idxCaption][1]),
                    captions[idxCaption][2]
                )
                vtt.captions.append(caption)

            # Increment offset
            offsetMilliSeconds=offsetMilliSeconds+captions[len(captions)-1][1]

        #print(captions[0][0])
        #print(millisec_to_hms(captions[0][0]))
        #captions = webvtt.read(vtt_file_name)
        #print(captions[0].start)
        #captions[0].start = '00:00:01.250'
        #print(captions[0].start)

        vtt.save('global_captions.vtt')
    st.text("General vtt file generated.")
    
    return global_vtt_file