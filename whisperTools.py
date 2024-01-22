import os
import math
from pydub import AudioSegment
import openai
import whisper
from pyannote_whisper.utils import diarize_text
from pyannote.core import Segment
import datetime
import json
import streamlit as st
import fileTools

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

def diarize_wav(output_path,wav_file_name,diarization_result,whisperViaApi=True):

    # Whisper is an Automatic Speech Recognition (ASR) developped by OpenIA
    # https://github.com/openai/whisper
    chunks_total =chunk_wav_file(output_path,wav_file_name)
    
    # load entire file
    # Auto Chuncking with opena ai audio transcribe
    # https://platform.openai.com/docs/guides/speech-to-text
    #my_bar = st.progress(1, text="Processing chunks")

    final_result =[]
    offset=0

    for x in range(chunks_total):
        file_to_process= str(x) + "-" + wav_file_name
        fullPath_file_to_process=output_path+"/" + file_to_process

        print(f" Processing openAI transcribe for : {fullPath_file_to_process}")
        audio_file = open(fullPath_file_to_process, "rb")
        # https://github.com/openai-php/client/issues/59
        # Difference response formats
        asr_result=""
        if(whisperViaApi):
            print(f" => Processing via OpenAI API")
            asr_result = openai.Audio.transcribe(
                model="whisper-1", 
                file=audio_file,
                language="fr",
                response_format="verbose_json"
                )
        else:
            print(f" => Processing via OpenAI local model")            
            import settings
            isStreamlitDeploy=  settings.isStreamlitCloudVersion
            modelSize="small"
            #modelSize="large"
            if(isStreamlitDeploy):
                modelSize="large"

            print(f"- Whisper model size => {modelSize}")    
            model = whisper.load_model(modelSize)
            asr_result = model.transcribe(fullPath_file_to_process, language='fr')

        fileTools.writeJsonFile(fullPath_file_to_process+".asr_result" ,asr_result)

        st.success(f' {str(x+1)}/{str((chunks_total))} - Transcribe chunk: {wav_file_name}', icon="✅") 

        if (final_result is None or len(final_result)==0):
            final_result = diarize_text(asr_result, diarization_result)
            # Récupération de l'offset (end du dernier élément)
            offset=final_result[len(final_result)-1][0].end
        else:
            final_result2 = diarize_text(asr_result, diarization_result)
            for elmt in final_result2:
               # Segment // Speaker // Content  
               newElmt=(Segment(elmt[0].start+offset,elmt[0].end+offset), elmt[1], elmt[2])
               final_result.append(newElmt)

            offset=offset+final_result2[len(final_result2)-1][0].end

    return final_result

# ################
# Not used for the moment
# ################
def whisper_segments_to_vtt_list(result_segments):
  """
  This function iterates through all whisper
  segements to format them into List with start time / end time / text.
  """
  data = "WEBVTT\n\n"
  for idx, segment in enumerate(result_segments):
    num = idx + 1
    data+= f"{num}\n"
    start_ = datetime.timedelta(seconds=segment.get('start'))
    start_ = timedelta_to_videotime(str(start_))
    end_ = datetime.timedelta(seconds=segment.get('end'))
    end_ = timedelta_to_videotime(str(end_))
    data += f"{start_} --> {end_}\n"
    text = segment.get('text').strip()
    data += f"{text}\n\n"
  return data

def whisper_segments_to_vtt_data(result_segments):
  """
  This function iterates through all whisper
  segements to format them into List with start time / end time / text.
  """
  data = "WEBVTT\n\n"
  for idx, segment in enumerate(result_segments):
    num = idx + 1
    data+= f"{num}\n"
    start_ = datetime.timedelta(seconds=segment.get('start'))
    start_ = timedelta_to_videotime(str(start_))
    end_ = datetime.timedelta(seconds=segment.get('end'))
    end_ = timedelta_to_videotime(str(end_))
    data += f"{start_} --> {end_}\n"
    text = segment.get('text').strip()
    data += f"{text}\n\n"
  return data

def timedelta_to_videotime(delta):
    
  """
  Here's a janky way to format a 
  datetime.timedelta to match 
  the format of vtt timecodes. 
  """
  parts = delta.split(":")
  if len(parts[0]) == 1:
    parts[0] = f"0{parts[0]}"
  new_data = ":".join(parts)
  parts2 = new_data.split(".")
  if len(parts2) == 1:
    parts2.append("000")
  elif len(parts2) == 2:
    parts2[1] = parts2[1][:2]
  final_data = ".".join(parts2)
  return final_data

# ################
# / Not used for the moment
# ################
