import pandas as pd
import os
import glob
import re
import PyPDF2
from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm
from google.colab import drive
import shutil

# 1. Improved Google Drive mounting with better error handling
def mount_drive():
    try:
        # Check if drive is already mounted
        if os.path.exists('/content/drive') and os.path.isdir('/content/drive/My Drive'):
            print("Google Drive semble déjà monté.")
            return True
        
        # Try to mount drive
        drive.mount('/content/drive')
        print("Google Drive monté avec succès")
        return True
    except Exception as e:
        print(f"Erreur lors du montage de Google Drive: {str(e)}")
        return False

# Télécharger les ressources NLTK nécessaires
nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Erreur lors de l'extraction du texte de {pdf_path}: {e}")
    
    # Nettoyer le texte (supprimer les espaces multiples, etc.)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_text_from_txt(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Essayer avec une autre encodage si utf-8 échoue
        try:
            with open(txt_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Erreur lors de la lecture de {txt_path}: {e}")
            return ""

def chunk_text(text, max_chunk_size=2000, overlap=200):
    """
    Découpe un long texte en chunks plus petits tout en préservant les phrases.
    
    Args:
        text (str): Texte à découper
        max_chunk_size (int): Taille maximale de chaque chunk en caractères
        overlap (int): Chevauchement entre les chunks en caractères
        
    Returns:
        list: Liste des chunks de texte
    """
    # Diviser le texte en phrases
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Si l'ajout de cette phrase dépasse la taille maximale du chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Garder une partie du chunk précédent pour le contexte (overlap)
            words = current_chunk.split()
            # S'assurer que l'overlap ne dépasse pas la taille du chunk précédent
            overlap_size = min(len(current_chunk), overlap)
            current_chunk = current_chunk[-overlap_size:] if overlap_size > 0 else ""
        
        current_chunk += " " + sentence
    
    # Ajouter le dernier chunk s'il n'est pas vide
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_summary(text, ratio=0.2, min_length=50, max_length=150):
    """
    Génère un résumé extractif simple basé sur les premières phrases du texte.
    Dans un cas réel, vous utiliseriez un modèle de résumé plus sophistiqué.
    
    Args:
        text (str): Texte à résumer
        ratio (float): Ratio du texte original à conserver pour le résumé
        min_length (int): Longueur minimale du résumé en caractères
        max_length (int): Longueur maximale du résumé en caractères
        
    Returns:
        str: Résumé du texte
    """
    # Diviser le texte en phrases
    sentences = sent_tokenize(text)
    
    # Pour un résumé simple, prendre les premières phrases
    num_sentences = max(1, min(int(len(sentences) * ratio), 3))
    
    # Prendre les premières phrases comme résumé
    summary = " ".join(sentences[:num_sentences])
    
    # Tronquer si nécessaire pour respecter la longueur maximale
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit(' ', 1)[0] + "..."
    
    # S'assurer que le résumé a une longueur minimale
    if len(summary) < min_length and len(sentences) > num_sentences:
        additional_sentences = min(1, len(sentences) - num_sentences)
        summary = " ".join(sentences[:num_sentences + additional_sentences])
    
    return summary.strip()

# 2. Improved file search function with better directory verification
def create_dataset_from_files(input_dir, train_ratio=0.8, max_samples=None):
    """
    Crée un dataset à partir des fichiers texte et PDF d'un répertoire.
    
    Args:
        input_dir (str): Répertoire contenant les fichiers source
        train_ratio (float): Ratio de données à utiliser pour l'entraînement
        max_samples (int): Nombre maximum d'échantillons à inclure (None pour tous)
        
    Returns:
        tuple: (train_data, val_data) où chaque élément est un dictionnaire avec des clés 'text' et 'summary'
    """
    # Vérification que le répertoire existe
    if not os.path.exists(input_dir):
        print(f"Le répertoire d'entrée n'existe pas: {input_dir}")
        print(f"Répertoires disponibles dans /content/drive: {os.listdir('/content/drive') if os.path.exists('/content/drive') else 'Drive non monté'}")
        return {'text': [], 'summary': []}, {'text': [], 'summary': []}
    
    # Rechercher tous les fichiers texte et PDF
    txt_files = glob.glob(os.path.join(input_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True)
    
    all_files = txt_files + pdf_files
    print(f"Fichiers trouvés: {len(all_files)} ({len(txt_files)} TXT, {len(pdf_files)} PDF)")
    
    # 3. Ajouter un exemple de fichier si aucun fichier n'est trouvé
    if len(all_files) == 0:
        print("Aucun fichier trouvé. Création d'un exemple...")
        example_file = os.path.join(input_dir, "exemple.txt")
        example_text = """
        La situation économique en France montre des signes d'amélioration après la crise sanitaire. 
        Les indicateurs économiques sont en hausse, notamment dans le secteur des services et du tourisme. 
        Cependant, l'inflation reste une préoccupation majeure pour les ménages et les entreprises. 
        La Banque de France prévoit une croissance de 1,2% pour l'année en cours, un chiffre revu à la 
        baisse par rapport aux estimations précédentes.
        
        Une découverte archéologique majeure a été réalisée près de Lyon. Des chercheurs ont mis au jour 
        les vestiges d'une villa romaine datant du IIe siècle après J.-C. Le site comprend des mosaïques 
        exceptionnellement bien conservées, un système de chauffage par le sol et plusieurs statues en marbre. 
        Cette découverte permet de mieux comprendre l'organisation sociale et la vie quotidienne à l'époque 
        gallo-romaine. Le site sera ouvert au public après la fin des fouilles et la mise en place de mesures 
        de conservation.
        """
        try:
            with open(example_file, "w", encoding="utf-8") as f:
                f.write(example_text)
            print(f"Exemple de fichier créé: {example_file}")
            all_files = [example_file]
        except Exception as e:
            print(f"Erreur lors de la création du fichier exemple: {e}")
    
    all_texts = []
    
    # Extraire le texte de chaque fichier
    for file_path in tqdm(all_files, desc="Extraction des textes"):
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:  # .txt
            text = extract_text_from_txt(file_path)
        
        if text.strip():
            # Diviser les longs textes en chunks
            chunks = chunk_text(text)
            all_texts.extend(chunks)
    
    # Limiter le nombre d'échantillons si spécifié
    if max_samples and len(all_texts) > max_samples:
        all_texts = all_texts[:max_samples]
    
    print(f"Total de chunks de texte extraits: {len(all_texts)}")
    
    # Générer des résumés pour chaque chunk de texte
    data = {
        'text': [],
        'summary': []
    }
    
    for text_chunk in tqdm(all_texts, desc="Génération des résumés"):
        if len(text_chunk) > 100:  # Ignorer les chunks trop courts
            summary = generate_summary(text_chunk)
            data['text'].append(text_chunk)
            data['summary'].append(summary)
    
    # Diviser en ensembles d'entraînement et de validation
    split_idx = int(len(data['text']) * train_ratio)
    
    train_data = {
        'text': data['text'][:split_idx],
        'summary': data['summary'][:split_idx]
    }
    
    val_data = {
        'text': data['text'][split_idx:],
        'summary': data['summary'][split_idx:]
    }
    
    return train_data, val_data

def save_datasets(train_data, val_data, output_dir='.'):
    """
    Sauvegarde les datasets au format CSV.
    
    Args:
        train_data (dict): Données d'entraînement (textes et résumés)
        val_data (dict): Données de validation (textes et résumés)
        output_dir (str): Répertoire de sortie
    
    Returns:
        tuple: (chemin_train, chemin_val) chemins des fichiers CSV créés
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_fr.csv')
    val_path = os.path.join(output_dir, 'val_fr.csv')
    
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Fichier train_fr.csv créé avec {len(train_df)} exemples")
    print(f"Fichier val_fr.csv créé avec {len(val_df)} exemples")
    
    return train_path, val_path

def main(input_directory="/content/drive/My Drive/data", 
         output_directory="/content/drive/My Drive/data_csv",
         train_ratio=0.8,
         max_samples=None):
    """
    Fonction principale pour extraire le texte et créer les datasets.
    
    Args:
        input_directory (str): Chemin vers le répertoire contenant les fichiers source sur Google Drive
        output_directory (str): Chemin vers le répertoire de sortie sur Google Drive
        train_ratio (float): Proportion des données pour l'entraînement
        max_samples (int): Nombre maximum d'échantillons à inclure
    """
    # Monter Google Drive
    drive_mounted = mount_drive()
    if not drive_mounted:
        print("Impossible de monter Google Drive. Utilisation des chemins locaux.")
        input_directory = "./input_data"
        output_directory = "./output_data"
    
    # 4. Afficher la structure de Google Drive pour aider au débogage
    if drive_mounted:
        print("Structure de Google Drive:")
        if os.path.exists('/content/drive/My Drive'):
            print("Contenu de /content/drive/My Drive:", os.listdir('/content/drive/My Drive'))
        else:
            print("Le dossier My Drive n'existe pas dans /content/drive")
            if os.path.exists('/content/drive'):
                print("Contenu de /content/drive:", os.listdir('/content/drive'))
    
    # Créer les répertoires s'ils n'existent pas
    os.makedirs(input_directory, exist_ok=True)
    os.makedirs(output_directory, exist_ok=True)
    
    print(f"Utilisation du répertoire d'entrée: {input_directory}")
    print(f"Utilisation du répertoire de sortie: {output_directory}")
    
    # Extraire le texte et créer les datasets
    train_data, val_data = create_dataset_from_files(
        input_directory, 
        train_ratio=train_ratio,
        max_samples=max_samples
    )
    
    # Sauvegarder les datasets au format CSV
    train_path, val_path = save_datasets(train_data, val_data, output_directory)
    
    print(f"Datasets créés et sauvegardés dans {train_path} et {val_path}")
    return train_path, val_path

if __name__ == "__main__":
    main()