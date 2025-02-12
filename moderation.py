import os, boto3
import cv2
import time
import urllib.request
import json

from dotenv import load_dotenv
from matplotlib import pyplot as plt

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


def check_filetype(filename):
    """
    Détermine le type de fichier en fonction de son extension.

    Cette fonction prend un nom de fichier en entrée, extrait son extension et détermine
    le type de fichier (par exemple, image, vidéo). Si l'extension du fichier est reconnue comme un format
    d'image courant (jpg, png, tiff, svg) ou un format de vidéo courant (mp4, avi, mkv), elle attribue
    le type correspondant. Sinon, le type de fichier est défini sur None.

    Paramètres :
    - filename (str) : Le chemin vers le fichier incluant le nom de fichier.

    Retourne :
    - str ou None : Le type de fichier déterminé ('image', 'vidéo') ou None si le type de fichier
      n'est pas reconnu.

    Exemple :
    >>> check_filetype("/chemin/vers/image.jpg")
    'image'
    >>> check_filetype("/chemin/vers/video.mp4")
    'vidéo'
    >>> check_filetype("/chemin/vers/fichierinconnu.xyz")
    None
    """

    # Extrait le nom de base du fichier à partir du chemin de fichier fourni.
    file_basename = os.path.basename(filename)

    # Sépare le nom de base sur le point et prend la dernière partie comme extension.
    extension = file_basename.split(".")[-1]

    # Détermine le type de fichier en fonction de l'extension.
    if extension in ["jpg", "png", "tiff", "svg"]:
        filetype = "image"
    elif extension in ["mp4", "avi", "mkv"]:
        filetype = "video"
    else:
        filetype = None

    # Enregistre le type de fichier détecté.
    print(f"[INFO] : Le fichier {file_basename} est de type : {filetype}")
    
    return filetype


def extract_frame_video(video_path, frame_id):
    """
    Extrait une image spécifique d'une vidéo.

    Cette fonction utilise OpenCV pour ouvrir une vidéo à partir du chemin spécifié et extrait une image
    particulière en fonction de son ID. L'ID de l'image correspond à l'ordre de l'image dans la vidéo, en commençant
    par 0 pour la première image. Si l'extraction réussit, l'image est retournée sous forme d'un tableau Numpy.

    Paramètres :
    - video_path (str) : Le chemin vers le fichier vidéo d'où extraire l'image.
    - frame_id (int) : L'identifiant (ID) de l'image à extraire.

    Retourne :
    - ndarray ou None : L'image extraite (un tableau Numpy) si l'extraction est réussie,
      sinon `None`.

    Exemple :
    >>> image = extract_frame_video("/chemin/vers/video.mp4", 150)
    >>> type(image)
    <class 'numpy.ndarray'>
    """

    # Ouvre la vidéo à partir du chemin fourni.
    video = cv2.VideoCapture(video_path)

    # Positionne le lecteur vidéo sur l'image spécifiée par frame_id.
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    # Lit l'image actuelle.
    ret, image = video.read()

    # Si la lecture réussit (ret est True), retourne l'image.
    # Sinon, retourne None.
    return image if ret else None

def get_aws_session():
    """
    Crée et retourne une session AWS.

    Cette fonction charge les variables d'environnement depuis un fichier .env situé dans le répertoire
    courant ou les parents de celui-ci, récupère les clés d'accès AWS (`ACCESS_KEY` et `SECRET_KEY`),
    et initialise une session AWS avec ces identifiants ainsi qu'avec une région spécifiée (dans cet exemple,
    'us-east-1'). Elle est particulièrement utile pour configurer une session AWS de manière sécurisée sans
    hardcoder les clés d'accès dans le code.

    Retourne :
    - Session : Un objet session de boto3 configuré avec les clés d'accès et la région AWS.

    Exemple d'utilisation :
    >>> session_aws = get_aws_session()
    >>> type(session_aws)
    <class 'boto3.session.Session'>
    """

    # Charge les variables d'environnement depuis .env.
    load_dotenv()

    # Crée une session AWS avec les clés d'accès et la région définies dans les variables d'environnement.
    aws_session = boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),        # Récupère l'ID de clé d'accès depuis les variables d'environnement.
        aws_secret_access_key=os.getenv("SECRET_KEY"),    # Récupère la clé d'accès secrète depuis les variables d'environnement.
        region_name="us-east-1"                           # Spécifie la région AWS à utiliser.
    )
    
    # Retourne l'objet session créé.
    return aws_session

def moderate_image(image_path, aws_service):
    """
    Détecte du contenu nécessitant une modération dans une image en utilisant un service AWS spécifié.

    Cette fonction ouvre une image depuis un chemin donné, puis utilise le service AWS (comme Amazon Rekognition)
    pour détecter les contenus potentiellement inappropriés ou sensibles (comme la nudité, la violence, etc.).
    Elle collecte et retourne une liste des étiquettes de modération identifiées pour cette image.

    Paramètres :
    - image_path (str) : Le chemin vers l'image à analyser.
    - aws_service (object) : Un objet de service AWS configuré, capable de réaliser des opérations de détection
      de contenu nécessitant une modération (par exemple, un client Amazon Rekognition).

    Retourne :
    - list[str] : Une liste des noms des étiquettes de modération détectées pour l'image.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> moderate_image("/chemin/vers/image.jpg", aws_rekognition_client)
    ['Nudity', 'Explicit Violence']
    """
    
    with open(image_path, 'rb') as image:

        response = aws_service.detect_labels(
            Image={
                'Bytes': image.read()
            },
            MaxLabels=100,
            MinConfidence=50
        )

    objects_list = [dic["Name"] for dic in response["Labels"]][:3]

    return objects_list


def get_text_from_speech(filename, aws_service, job_name, bucket_name):
    """
    Convertit de la parole en texte en utilisant AWS Transcribe.

    Cette fonction téléverse un fichier audio spécifié dans un seau S3, démarre un travail de transcription avec AWS Transcribe,
    attend que le travail soit terminé, et récupère le texte transcrit.

    Paramètres :
    - filename (str) : Chemin local vers le fichier audio à transcrire.
    - aws_service (object) : Client AWS Transcribe configuré.
    - job_name (str) : Nom unique pour le travail de transcription.
    - bucket_name (str) : Nom du seau S3 où le fichier audio est stocké.

    Retourne :
    - str : Le texte transcrit du fichier audio.

    Prérequis :
    - Le fichier audio doit déjà être téléversé dans le seau S3 spécifié.
    """

    language_code = "fr-FR"

    # 1️⃣ Upload du fichier audio sur S3
    try:
        with open(filename, "rb") as f:
            AWS_SESSION = get_aws_session()
            s3 = AWS_SESSION.client('s3')
            s3.upload_file(filename, bucket_name, os.path.basename(filename))
        print(f"✅ Fichier {filename} uploadé sur S3 ({bucket_name})")
    except Exception as e:
        print(f"❌ Erreur lors de l'upload : {e}")
        return None

    # URL du fichier audio sur S3
    file_uri = f"s3://{bucket_name}/{os.path.basename(filename)}"

    # 2️⃣ Lancer le job de transcription
    try:
        aws_service.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': file_uri},
            MediaFormat=filename.split('.')[-1],  # Détecte le format (mp3, wav, etc.)
            LanguageCode=language_code
        )
        print(f"🎙️ Job de transcription '{job_name}' lancé...")
    except Exception as e:
        print(f"❌ Erreur lors du lancement de la transcription : {e}")
        return None

    # 3️⃣ Attente de la fin du job
    while True:
        response = aws_service.get_transcription_job(TranscriptionJobName=job_name)
        status = response['TranscriptionJob']['TranscriptionJobStatus']
        if status in ['COMPLETED', 'FAILED']:
            break
        print("⏳ En attente de la transcription...")
        time.sleep(5)

    if status == 'FAILED':
        print("❌ La transcription a échoué.")
        return None

    # 4️⃣ Récupération du texte transcrit
    transcript_url = response['TranscriptionJob']['Transcript']['TranscriptFileUri']
    print(f"✅ Transcription terminée ! Résultat disponible ici : {transcript_url}")

    # Télécharger le texte transcrit
    try:
        # Utilisation de urllib.request pour télécharger le fichier
        with urllib.request.urlopen(transcript_url) as response:
            transcript_data = json.loads(response.read().decode('utf-8'))
        transcript_text = transcript_data['results']['transcripts'][0]['transcript']
        return transcript_text
    except Exception as e:
        print(f"❌ Erreur lors de la récupération du texte : {e}")
        return None

def clean_text(raw_text):
    """
    Nettoie un texte en retirant les mots vides et en normalisant les mots en minuscules.

    Cette fonction prend un texte brut en entrée, tokenise le texte pour séparer les mots,
    convertit les mots en minuscules, et retire les mots vides (stop words) en français. Les mots vides
    supplémentaires peuvent être ajoutés à la liste. Le texte résultant contient uniquement les mots significatifs
    en minuscules.

    Paramètres :
    - raw_text (str) : Le texte brut à nettoyer.

    Retourne :
    - str : Le texte nettoyé, sans mots vides et en minuscules.

    Exemple d'utilisation :
    >>> texte_brut = "Ceci est un exemple de texte à nettoyer."
    >>> clean_text(texte_brut)
    'exemple texte nettoyer'
    """

    # Initialisation du tokenizer : il va découper le texte en mots en prenant en compte uniquement les lettres et chiffres
    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenisation du texte
    words = tokenizer.tokenize(raw_text)

    # Liste des mots vides en français (stop words) de NLTK
    stop_words = set(stopwords.words('french'))

    # Liste personnalisée de mots vides à exclure, on peut modifier cette liste
    custom_stop_words = {'ceci' }  # Ajouter des mots que tu veux exclure
    stop_words.update(custom_stop_words)  # Ajouter ces mots à la liste des stopwords

    # Filtrer les mots : on garde uniquement ceux qui ne sont pas dans la liste des mots vides
    cleaned_words = [word.lower() for word in words if word.lower() not in stop_words]

    # Rejoindre les mots pour former le texte nettoyé
    cleaned_text = ' '.join(cleaned_words)
    
    return cleaned_text

def extract_keyphrases(text, aws_service):
    """
    Extrait les expressions clés d'un texte et retourne les 10 expressions les plus pertinentes comme hashtags.

    Cette fonction utilise un service AWS, tel que Amazon Comprehend, pour détecter les expressions clés dans
    un texte donné. Elle trie ces expressions par leur score de pertinence fourni par AWS et retourne les 10
    expressions clés les plus pertinentes sous forme de hashtags.

    Paramètres :
    - text (str) : Le texte duquel extraire les expressions clés.
    - aws_service (object) : Un objet de service AWS configuré pour détecter les expressions clés.

    Retourne :
    - list[str] : Une liste des 10 hashtags les plus pertinents basés sur les expressions clés du texte.

    Exemple d'utilisation :
    >>> aws_comprehend_client = boto3.client('comprehend', region_name='us-east-1')
    >>> extract_keyphrases("Ceci est un exemple de texte.", aws_comprehend_client)
    ['#exemple', '#texte']
    """

    response =aws_service.detect_key_phrases(
        Text=text,
        LanguageCode='fr'
    )

    key_phrases = response['KeyPhrases']
    list_keyword = key_phrases[0]["Text"].split()
    hashtagged_keywords = ['#' + word for word in list_keyword]

    return hashtagged_keywords


def detect_objects(image_path, aws_service):
    """
    Détecte les objets dans une image en utilisant Amazon Rekognition.

    Cette fonction ouvre une image depuis un chemin spécifié, utilise un service AWS (Amazon Rekognition) pour
    détecter les objets présents dans l'image avec une confiance minimale de 50%, et retourne les noms des 10
    objets les plus pertinents détectés.

    Paramètres :
    - image_path (str) : Le chemin vers l'image à analyser.
    - aws_service (object) : Un client AWS Rekognition configuré.

    Retourne :
    - list[str] : Une liste contenant les noms des 10 premiers objets détectés dans l'image.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> detect_objects("/chemin/vers/image.jpg", aws_rekognition_client)
    ['Voiture', 'Arbre', 'Personne']
    """

    with open(image_path, 'rb') as image:

        response = aws_service.detect_labels(
            Image={
                'Bytes': image.read()
            },
            MaxLabels=10,
            MinConfidence=50
        )

    objects_list = [dic["Name"] for dic in response["Labels"]]

    return objects_list


def detect_celebrities(image_path, aws_service):
    """
    Identifie les célébrités dans une image en utilisant le service Amazon Rekognition.

    Cette fonction ouvre une image depuis un chemin donné et utilise le service AWS Rekognition pour reconnaître les
    célébrités présentes dans l'image. Elle retourne une liste contenant les noms des célébrités identifiées, limitée
    aux 10 premiers résultats pour simplifier l'output.

    Paramètres :
    - image_path (str) : Le chemin vers l'image dans laquelle détecter les célébrités.
    - aws_service (object) : Un client AWS Rekognition configuré.

    Retourne :
    - list[str] : Une liste des noms des célébrités identifiées dans l'image, jusqu'à un maximum de 10.

    Exemple d'utilisation :
    >>> aws_rekognition_client = boto3.client('rekognition', region_name='us-east-1')
    >>> detect_celebrities("/chemin/vers/limage.jpg", aws_rekognition_client)
    ['Leonardo DiCaprio', 'Kate Winslet']
    """

    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    response = aws_service.recognize_celebrities(
        Image={'Bytes': image_bytes}
    )

    # Récupération des noms des célébrités détectées
    celebrities = [celeb["Name"] for celeb in response["CelebrityFaces"]]

    return celebrities

def detect_emotions(image_path, aws_service):
    """
    Détecte les émotions sur les visages présents dans une image en utilisant Amazon Rekognition.
    
    Cette fonction analyse une image pour détecter les visages et leurs émotions associées.
    Pour chaque visage, elle retourne les émotions détectées avec leur niveau de confiance.
    
    Paramètres :
    - image_path (str) : Chemin vers l'image à analyser
    - aws_service (boto3.client) : Client AWS Rekognition configuré
    
    Retourne :
    - list[dict] : Liste des visages détectés avec leurs émotions
                  Format: [
                      {
                          'BoundingBox': dict,
                          'Emotions': [
                              {
                                  'Type': str,  # HAPPY, SAD, ANGRY, CONFUSED, etc.
                                  'Confidence': float
                              },
                              ...
                          ],
                          'AgeRange': {'Low': int, 'High': int},
                          'Gender': {'Value': str, 'Confidence': float}
                      },
                      ...
                  ]
    
    Exemple :
    >>> rekognition = boto3.client('rekognition')
    >>> emotions = detect_emotions("./photo.jpg", rekognition)
    >>> for face in emotions:
    ...     print(f"Émotions détectées : {face['Emotions']}")
    """

    # Lecture de l’image en binaire
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # Appel à Amazon Rekognition
    response = aws_service.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL']  # Permet d'obtenir toutes les informations, y compris les émotions
    )

    # Extraction des informations
    faces_data = []
    for face in response['FaceDetails']:
        face_info = {
            'BoundingBox': face['BoundingBox'],
            'Emotions': face['Emotions'],
            'AgeRange': face['AgeRange'],
            'Gender': face['Gender']
        }
        faces_data.append(face_info)

    return faces_data


def summarize_emotions(faces_info):
    """
    Résume les émotions détectées sur tous les visages d'une image.
    
    Cette fonction agrège les émotions de tous les visages et calcule les émotions
    dominantes dans l'image.
    
    Paramètres :
    - faces_info (list[dict]) : Liste des informations des visages détectés
    
    Retourne :
    - dict : Résumé des émotions dominantes et statistiques
    
    Exemple :
    >>> emotions = detect_emotions("./group_photo.jpg", rekognition)
    >>> summary = summarize_emotions(emotions)
    >>> print(f"Émotion dominante : {summary['dominant_emotion']}")
    """

    if not faces_info:
        return {
            "total_faces": 0,
            "dominant_emotion": None,
            "emotion_stats": {},
            "age_stats": {"min": None, "max": None, "average": None},
            "gender_distribution": {}
        }

    emotion_count = {}
    emotion_confidence = {}
    ages = []
    gender_count = {}

    for face in faces_info:
        # Collecte des genres
        gender = face["Gender"]["Value"]
        gender_count[gender] = gender_count.get(gender, 0) + 1

        # Collecte des âges
        age_avg = (face["AgeRange"]["Low"] + face["AgeRange"]["High"]) / 2
        ages.append(age_avg)

        # Collecte des émotions avec confiance > 50%
        for emotion in face["Emotions"]:
            if emotion["Confidence"] > 50:
                emotion_type = emotion["Type"]
                if emotion_type not in emotion_count:
                    emotion_count[emotion_type] = 0
                    emotion_confidence[emotion_type] = 0
                emotion_count[emotion_type] += 1
                emotion_confidence[emotion_type] += emotion["Confidence"]

    # Déterminer l'émotion dominante (celle avec la meilleure confiance moyenne)
    dominant_emotion = None
    highest_avg_confidence = 0

    for emotion in emotion_count:
        avg_confidence = emotion_confidence[emotion] / emotion_count[emotion]
        if avg_confidence > highest_avg_confidence:
            dominant_emotion = emotion
            highest_avg_confidence = avg_confidence

        # Mise à jour des statistiques des émotions
        emotion_confidence[emotion] = round(avg_confidence, 2)

    # Calcul des statistiques d'âge
    age_stats = {
        "min": min(ages),
        "max": max(ages),
        "average": sum(ages) / len(ages)
    }

    return {
        "total_faces": len(faces_info),
        "dominant_emotion": dominant_emotion,
        "emotion_stats": {
            emotion: {
                "count": emotion_count[emotion],
                "average_confidence": emotion_confidence[emotion]
            } for emotion in emotion_count
        },
        "age_stats": age_stats,
        "gender_distribution": gender_count
    }

def process_media(media_file, rekognition, transcribe, comprehend, bucket_name):
    """
    Traite un fichier multimédia (image ou vidéo) pour modérer le contenu, détecter des objets/célébrités,
    transcrire le discours et extraire des expressions clés.

    Selon le type de fichier, cette fonction applique une chaîne de traitement appropriée en utilisant différents
    services AWS. Pour les images, elle modère le contenu, détecte des objets, émotions faciales et des célébrités. Pour les vidéos,
    elle extrait une image, modère le contenu, téléverse la vidéo sur S3, transcrit le discours en texte, nettoie le texte,
    et extrait des expressions clés.

    Paramètres :
    - media_file (str) : Chemin vers le fichier multimédia à traiter.
    - rekognition (object) : Client AWS Rekognition configuré.
    - transcribe (object) : Client AWS Transcribe configuré.
    - comprehend (object) : Client AWS Comprehend configuré.
    - bucket_name (str) : Nom du seau S3 pour stocker les fichiers vidéo.

    Retourne :
    - dict : Dictionnaire contenant des hashtags pour les images ou des sous-titres et hashtags pour les vidéos.
    """
    translate = get_aws_session().client('translate', region_name='us-east-1')
    file_type = check_filetype(media_file)

    inappropriate_labels = ["Pornography", "Fighting", "Weapon", "Protest", "Explicit", "Porn", "XXX", "Nudity", "War", "Suggestive Content", 
        "Violence", "Assault", "Torture", "Drug", "Hate", "Abuse", "Hate Scpeech", "Suicide", "Terrorism"]

    if file_type == "image":
        data = {
            "moderation_list" : moderate_image(image_path=media_file, aws_service=rekognition),
            "object_list" : detect_objects(image_path=media_file, aws_service=rekognition),
            "emotion_list" : summarize_emotions(detect_emotions(image_path=media_file, aws_service=rekognition)),
            "celebrity_list" : detect_celebrities(image_path=media_file, aws_service=rekognition)
        }

        inappropriate = []
        for key, value in data.items():
            if isinstance(value, list):
                for label in value:
                    if label in inappropriate_labels:
                        inappropriate.append(label)
        
        if(inappropriate):
            return {
                "error" : f"""Le contenu que vous avez chargé est inapproprié. 
                    Contenu(s) détecté(s) : {', '.join(inappropriate)}""",
                'hashtag': ""
            }

        hashtag_list = ["#"+data["emotion_list"]["dominant_emotion"]]
        hashtag_list += ["#" + word for word in data["celebrity_list"] if word or not None]
        hashtag_list += ["#" + word for word in data["object_list"][:3]]
        return {
            "hashtag": hashtag_list
        }

    elif file_type == "video":
        print("File = video")

        text = get_text_from_speech(
            filename=media_file,
            aws_service=transcribe,
            job_name=f"transcription-{int(time.time())}",
            bucket_name=bucket_name
        )

        return {
            "sous-titres": text,
            "hashtag": extract_keyphrases(text=clean_text(text), aws_service=comprehend)
        }

if __name__ == "__main__":

    AWS_SESSION = get_aws_session()
    BUCKET_NAME = 'sdv-tp-socialmedia-transcribe'

    s3 = AWS_SESSION.client('s3')
    s3.create_bucket(Bucket='sdv-tp-socialmedia')


    # Tester le type de fichier
    TEST_VIDEO_FILE = "./assets/tuto_maquillage.mp4"
    TEST_IMAGE_FILE = "./assets/selfie_with_johnny-depp.png"
    # video = check_filetype(TEST_VIDEO_FILE)
    # image = check_filetype(TEST_IMAGE_FILE)


    # # Afficher la première frame de la vidéo
    # TEST_VIDEO_FILE = "./assets/tuto_jeux-video.mp4"
    # frame_video = extract_frame_video(TEST_VIDEO_FILE,99)
    # imgplot = plt.imshow(frame_video)
    # plt.show()

    TEST_IMAGE_FILE_1 = "./assets/haine.png"
    TEST_IMAGE_FILE_2 = "./assets/vulgaire.png"
    TEST_IMAGE_FILE_3 = "./assets/violence1.png"
    TEST_IMAGE_FILE_4 = "./assets/no-violence1.png"

    # print(moderate_image(TEST_IMAGE_FILE_1, AWS_SESSION.client('rekognition')))
    # quit()
    # print(moderate_image(TEST_IMAGE_FILE_2, AWS_SESSION))
    # print(moderate_image(TEST_IMAGE_FILE_3, AWS_SESSION))
    # print(moderate_image(TEST_IMAGE_FILE_4, AWS_SESSION))

    
    TEST_VIDEO_FILE = "./assets/tuto_coiffure.mp4"
    BUCKET_NAME = 'sdv-tp-socialmedia-transcribe'

    text=get_text_from_speech(
            filename=TEST_VIDEO_FILE,
            aws_service=AWS_SESSION.client('transcribe'),
            job_name=f"transcription-{int(time.time())}",
            bucket_name=BUCKET_NAME
        )
    print(text)
    # cleaned_text = clean_text(text)
    
    # print(extract_keyphrases(
    #     text=cleaned_text,
    #     aws_service=AWS_SESSION
    # ))

    TEST_IMAGE_FILE = "./assets/no-violence4.png"
    # print(detect_objects(TEST_IMAGE_FILE, AWS_SESSION))

    TEST_IMAGE_FILE_1 = "./assets/selfie_with_mariah-carey.png"
    TEST_IMAGE_FILE_2 = "./assets/selfie_with_johnny-depp.png"
    TEST_IMAGE_FILE_3 = "./assets/selfie_with_kanye-west.png"

    # print(detect_celebrities(TEST_IMAGE_FILE_1, AWS_SESSION))
    # print(detect_celebrities(TEST_IMAGE_FILE_2, AWS_SESSION))
    # print(detect_celebrities(TEST_IMAGE_FILE_3, AWS_SESSION))


    TEST_IMAGE_FILE_1 = "./assets/selfie_with_mariah-carey.png"

    # result = detect_emotions(TEST_IMAGE_FILE_1, AWS_SESSION)
    # for face in result:
    #     print(f"👤 Visage détecté : {face['BoundingBox']}")
    #     for emotion in face['Emotions']:
    #         print(f"   😃 {emotion['Type']} - {emotion['Confidence']:.2f}%")

    # Définir les chemins des images de test
    TEST_IMAGE_FILE_1 = "./assets/group_selfie_1.jpg"    # Premier selfie de groupe
    TEST_IMAGE_FILE_2 = "./assets/group_selfie_2.jpg"    # Deuxième selfie de groupe
    TEST_IMAGE_FILE_3 = "./assets/group_selfie_3.jpg"    # Troisième selfie de groupe
    TEST_IMAGE_FILE_4 = "./assets/group_selfie_4.jpg"    # Quatrième selfie de groupe

    # result1 = detect_emotions(TEST_IMAGE_FILE_1, AWS_SESSION)
    # result2 = detect_emotions(TEST_IMAGE_FILE_2, AWS_SESSION)
    # result3 = detect_emotions(TEST_IMAGE_FILE_3, AWS_SESSION)
    # result4 = detect_emotions(TEST_IMAGE_FILE_4, AWS_SESSION)

    # print(summarize_emotions(result1))
    # print(summarize_emotions(result2))
    # print(summarize_emotions(result3))
    # print(summarize_emotions(result4))




    # TEST_IMAGE_FILE_1 = "./assets/selfie_with_mariah-carey.png"
    # TEST_VIDEO_FILE_1 = "./assets/tuto_maquillage.mp4"
    # result = process_media(
    #     media_file=TEST_IMAGE_FILE_1,
    #     rekognition=AWS_SESSION.client('rekognition'),
    #     transcribe=AWS_SESSION.client('transcribe'),
    #     comprehend=AWS_SESSION.client('comprehend'),
    #     bucket_name=BUCKET_NAME
    # )
    # print(result)
















