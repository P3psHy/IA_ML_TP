import streamlit as st
import time
import tempfile
import moderation
import os, boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# ----------------------------------------------------------------------------- SIDEBAR ------------------------------------------------------------------------------------ #

st.sidebar.header("⚙️ Configuration")

credentials_env = st.sidebar.container(border=True)
credentials_env.subheader("🗝️ :orange[Credentials AWS]", divider="gray")

# Valeurs par défaut pour le formulaire
session_defaults = {
    "file_uploaded": False,
    "access_key": "",
    "secret_key": "",
    "upload_message": "",
    "upload_status": False,
    "show_uploader": False,
}

# Chargement de toutes les valeurs par défaut du formulaire
for key, default_value in session_defaults.items():
    st.session_state.setdefault(key, default_value)

# Création d'un fichier .env à partir des clés entrées par l'utilisateur
def save_credentials_to_env(access_key, secret_key):
    try:
        with open(".env", "w") as f:
            f.write(f"ACCESS_KEY={access_key}\n")
            f.write(f"SECRET_KEY={secret_key}\n")
    except Exception as e:
        st.session_state.upload_message(f"❌ Erreur lors de la sauvegarde des credentials: {e}")

def verify_aws_credentials():
    try:
        AWS_SESSION = moderation.get_aws_session()
        sts_client = AWS_SESSION.client("sts")

        # Vérification que les credentials sont valides        
        response = sts_client.get_caller_identity()

        return True, response

    except NoCredentialsError:
        return st.error("❌ Veuillez rentrer des credentials valides !")
    except PartialCredentialsError:
        return st.error("❌ Les credentials sont incomplets. Un access key et un secret key sont nécessaires.")
    except ClientError as e:
        error_message = str(e)
        if "InvalidClientTokenId" in error_message:
            return st.error("❌ Les credentials ne sont pas valides. Veuillez vérifier l'access key et/ou le secret key !")
        return st.error("❌ Erreur AWS Client : {error_message}")


# Chargement des clés de AWS à partir d'un fichier .env
def load_credentials_from_file(uploaded_file):
    try:
        content = uploaded_file.getvalue().decode("utf-8").strip()
        lines = content.split("\n")
        
        # Si le format du contenu du fichier n'est pas valide,
        if not lines[0].startswith("ACCESS_KEY=") or not lines[1].startswith("SECRET_KEY=") or len(lines) != 2:
            st.session_state.upload_message = "❌ Format invalide ! Le fichier doit contenir exactement un ACCESS_KEY et un SECRET_KEY."
            st.session_state.upload_status = False
            return
        
        # Si le format du contenu du fichier est valide,
        st.session_state.access_key = lines[0].split("=", 1)[1] # ACCESS_KEY
        st.session_state.secret_key = lines[1].split("=", 1)[1] # SECRET_KEY

        st.session_state.file_uploaded = True
        st.session_state.upload_message = "✅ Credentials chargés avec succès !"
        st.session_state.upload_status = True
        st.session_state.show_uploader = False 
        
        save_credentials_to_env(st.session_state.access_key, st.session_state.secret_key)

        st.rerun()

    except Exception as e:
        # Si problème de chargement,
        st.session_state.upload_message = f"❌ Erreur lors du chargement du fichier : {str(e)}"
        st.session_state.upload_status = False

# Si clic sur le bouton, alors afficher le chargeur de fichiers
if credentials_env.button("📁 Charger credentials depuis .env", type="primary"):
    st.session_state.show_uploader = True
    st.session_state.file_uploaded = False
    st.rerun()

if st.session_state.show_uploader and not st.session_state.file_uploaded:
    uploaded_file = credentials_env.file_uploader("Fichier env", type=["env"], accept_multiple_files=False, label_visibility="hidden")
    
    # Si un fichier a bien été chargé, alors commmencer le traitement.
    if uploaded_file is not None:
        load_credentials_from_file(uploaded_file)


SUCCESS_BG_COLOR = "#d4edda"
SUCCESS_TEXT_COLOR = "#155724"
ERROR_BG_COLOR = "#f8d7da"
ERROR_TEXT_COLOR = "#721c24"

# Affichage message de succès ou d'erreur lors du chargement du fichier
if st.session_state.upload_message:
    bg_color = SUCCESS_BG_COLOR if st.session_state.upload_status else ERROR_BG_COLOR
    text_color = SUCCESS_TEXT_COLOR if st.session_state.upload_status else ERROR_TEXT_COLOR

    credentials_env.markdown(
        f"""
        <div style='background-color: {bg_color}; color: {text_color}; padding: 10px; border-radius: 5px;'>
            {st.session_state.upload_message}
        </div>
        """, unsafe_allow_html=True
    )
    # Disparition du message au bout de 2 secondes
    time.sleep(2)
    st.session_state.upload_message = ""
    st.rerun()

access_key = credentials_env.text_input("Access Key", value=st.session_state.access_key, type="password")
secret_key = credentials_env.text_input("Secret Key", value=st.session_state.secret_key, type="password")


bucket_name = credentials_env.text_input("Nom du bucket S3", value="sdv-tp-socialmedia-erika", placeholder="exemple-bucket-sdv-2")



# ----------------------------------------------------------------------------- MAIN ------------------------------------------------------------------------------------ #

main = st.container()

main.title("✨ :blue[Content Moderator Pro] ✨")
main.divider()
main.write("")

description = main.container(border=True)
description.subheader("Analyser et modérer votre contenu en un clic !")
description.write("")

description.markdown("""
        **Content Moderator Pro** vous aide à analyser🕵️‍♂️ et modérer vos images📸 et vidéos🎥 facilement !\n
        Il vous suffit de :green[charger vos informations de connexion en quelques étapes], 
        et l'application se charge du reste 🪄. Elle est conçue pour vous offrir une façon 
        rapide et simple de gérer vos images et autres contenus en toute sécurité. 🔐""")

main.divider()

MAX_FILE_SIZE_Mo = 200
MAX_FILE_SIZE = MAX_FILE_SIZE_Mo = 200 * 1024 * 1024
ALLOWED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
ALLOWED_VIDEO_EXTENSIONS = ["mp4", "avi", "mov"]

with main:
    uploaded_content = st.file_uploader(
        "**:orange[Choisissez un fichier (image ou vidéo) :]**",
        type=ALLOWED_IMAGE_EXTENSIONS + ALLOWED_VIDEO_EXTENSIONS
    )

    if uploaded_content is not None:
        file_size = uploaded_content.size
        file_extension = uploaded_content.name.split(".")[-1].lower() 
        file_type = uploaded_content.type.split("/")[0]

        if file_size > MAX_FILE_SIZE:
            st.error(f"❌ Le fichier dépasse la limite de {MAX_FILE_SIZE_Mo}Mo. Veuillez charger un fichier moins lourd.")
        
        elif file_extension not in ALLOWED_IMAGE_EXTENSIONS + ALLOWED_VIDEO_EXTENSIONS:
            st.error("❌ Le type de fichier ne correspond pas à ce qui est attendu. Veuillez charger une image (jpg, png) ou une vidéo (mp4, avi, mov).")
        
        else:
            st.success("✅ Fichier chargé avec succès !")

            AWS_SESSION = moderation.get_aws_session()
            
            # Si les credentials renseignés sont valides,
            if verify_aws_credentials():
                rekognition=AWS_SESSION.client('rekognition')
                transcribe=AWS_SESSION.client('transcribe')
                comprehend=AWS_SESSION.client('comprehend')
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_content.name.split('.')[-1]}") as temp_file:
                    temp_file.write(uploaded_content.read())
                    temp_path = temp_file.name

                if file_type == "image":
                    moderation_values = moderation.process_media(temp_path, rekognition, transcribe, comprehend, bucket_name)

                    if('error' in moderation_values):
                        st.error(moderation_values["error"])
                    else:
                        st.image(uploaded_content, use_container_width=True)
                        hashtag_list = sum([value for key, value in moderation_values.items() if isinstance(value, list)], [])
                    
                elif file_type == "video":
                    st.video(uploaded_content, format="video/mp4")
                    video_path = AWS_SESSION.client('s3').upload_file(temp_path, bucket_name, f"uploads/{uploaded_content.name}")
                    hashtag_list = moderation.process_media(video_path, rekognition, transcribe, comprehend, bucket_name)
                else:
                    st.error("❌ Ce type de fichier n'est pas supporté. Veuillez réessayer !")
                
                hashtag = st.container()

                unique_hashtags = set(hashtag_list)

                nb_hashtags = len(unique_hashtags)
                num_columns = min(round(nb_hashtags / 2), 6)

                columns = st.columns(num_columns)

                unique_words_list = list(unique_hashtags)
                chunk_size = (nb_hashtags + num_columns - 1) // num_columns

                for i, col in enumerate(columns):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size
                    chunk = unique_words_list[start_index:end_index]
                    
                    with col:
                        for word in chunk:
                            st.markdown(f"""
                                <p style="font-size: 13px; background-color: #9DDCFC; color: #1E90FF; border-radius: 4px; padding: 10px; margin: 5px; height: 40px; white-space: no-wrap;">
                                    #{word}
                                </p>
                                """, unsafe_allow_html=True)     

        
                #with st.expander("See explanation"):
                #    st.write(f"""{moderation_values["sous-titres"]}""")
st.markdown(
    """
    <style>
        .stMain .block-container {
            max-width: 100% !important;
            padding-left: 4rem;
            padding-right: 4rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)