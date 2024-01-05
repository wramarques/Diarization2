import os
import pandas as pd
import streamlit as st
import base64
import re
import unicodedata
import json
from datetime import datetime

def CreateDir_if_not_exists(directoryName: str) -> bool:
    """
    Create directory if not exist.
    """
    path = os.path.dirname(os.path.realpath(__file__)) + '/'+ directoryName

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"create user workdir : {path}")
        return True

    print(f"User workdir exists : {path}")

    return False

def forceFileNameExtension (value :str, extension :str) -> str:
    """
    Force file name extension
    """
    ret = str(value)
    splitName=value.split('.')

    if(len(splitName)>1):
        return f"{splitName[0]}.{extension}"
    
    return value

def normalize_filename(value: str) -> str:
    """
    Normalizes string to ASCII, removes non-alpha characters, and converts spaces to underscores.
    """
    value = str(value)
    splitName=value.split('.')

    value = unicodedata.normalize("NFKD", splitName[0]).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip()
    value= re.sub(r"[-\s]+", "_", value) 
    return f"{value}.{splitName[1]}"

def save_uploadedfile(output_path,uploadedfile, fileNameDest ):
    with open(os.path.join(output_path,fileNameDest),"wb") as f:
        f.write(uploadedfile.getbuffer())
    st.success(f"Saved File: {fileNameDest} to working dir.")        

def writeTextFile(file_path,data):
    with open(file_path, "w") as text_file:
        text_file.write(str(data))

def writeJsonFile(file_path,data ):
    """
    Enregistre un objet JSON dans un fichier.

    :param data: L'objet Python à enregistrer (généralement un dictionnaire ou une liste).
    :param file_path: Le chemin du fichier où enregistrer le JSON.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Erreur lors de l'enregistrement dans le fichier: {e}")

def directory_to_dataframe(directory_path, pattern=""):
    """
    Crée un DataFrame avec les noms, tailles et extensions des fichiers d'un répertoire.

    :param directory_path: Chemin du répertoire.
    :return: DataFrame avec des informations de fichier.
    """
    files_data = []

    print(f"Scan {directory_path}") 

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # and 'transcript' in file_path 
        if os.path.isfile(file_path):

            if(pattern=="" or pattern in file_path):
                size = os.path.getsize(file_path)
                #name, extension = os.path.splitext(filename)
                name = filename
                extension = ""
                mod_time = os.path.getmtime(file_path)
                date_modified = datetime.fromtimestamp(mod_time)

                files_data.append({'Name': name, 'Size': size, 'Extension': extension, 'Path': file_path, 'Modification':date_modified})

    if(not (files_data is None)  and len(files_data)>0):
        # Trier par date de modification décroissante
        df= pd.DataFrame(files_data)
        df = df.sort_values(by='Modification', ascending=False)
        return df
    
    return None

def create_tabe_df(df, containerTable=st.sidebar):

    # Ajout d'un titre dans la sidebar
    #st.sidebar.title("Informations des Fichiers")

    # En-têtes
    header_cols = st.sidebar.columns([2, 1 ])
    header_cols[0].markdown("**Name**")
    header_cols[1].markdown(f'**Download**')

    # Création d'une mise en page en colonnes 🔽 📥 💾 ⬇️
    for _, row in df.iterrows():
        cols = containerTable.columns([2, 1])  # Ajustez les proportions des colonnes selon vos besoins

        cols[0].write(row['Name'])

        # Bouton de téléchargement
        # https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
        with open(row['Path'], 'rb') as file:
            cols[1].download_button(
                label="🔽",
                data=file,
                file_name=row['Name'],
                mime="application/octet-stream"
            )