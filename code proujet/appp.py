import streamlit as st
from langdetect import detect
import os
import tempfile

# Import custom modules
from Structured_Text_Summarizer import StructuredTextSummarizer
from structured_PDF_processor import StructuredPDFProcessor
from audio_processor import AudioProcessor  

# Initialize components
summarizer = StructuredTextSummarizer()
pdf_processor = StructuredPDFProcessor(summarizer=summarizer)
audio_processor = AudioProcessor(model_size="base")  # Initialisation du processeur audio

def main():
    # Page configuration avec un thème moderne
    st.set_page_config(
        page_title="Résumé Intelligent",
        layout="wide",
        page_icon="🧠",
        initial_sidebar_state="expanded"
    )
    
    # CSS personnalisé pour améliorer l'apparence
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #4776E6 0%, #8E54E9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    .info-box {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 1.5rem;
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # En-tête principal avec style amélioré
    st.markdown('<h1 class="main-header">🧠 Générateur de Résumés Multilingue</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Transformez vos textes, documents et audio en résumés concis et pertinents en quelques secondes.</p>', unsafe_allow_html=True)
    
    # Séparateur visuel
    st.markdown("---")
    
    # Layout principal en deux colonnes
    col_config, col_input = st.columns([1, 2])
    
    # Colonne de configuration à gauche
    with col_config:
        st.markdown("### ⚙️ Configuration")
        
        with st.container():
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            
            # Input method selection avec icônes
            st.subheader("Source du contenu")
            option = st.radio(
                "Choisissez une méthode d'entrée :",
                ["📄 PDF", "✏️ Texte", "🎤 Audio"],  # Ajout de l'option audio
                format_func=lambda x: x,
                horizontal=True
            )
            
            # Summary length selection avec visualisation
            st.subheader("Longueur du résumé")
            summary_length = st.select_slider(
                "Ajustez la longueur :",
                options=["short", "medium", "long"],
                value="medium"
            )
            
            # Visualisation graphique de la longueur
            length_display = {
                "short": "🟩⬜⬜⬜ Court (25%)",
                "medium": "🟩🟩⬜⬜ Moyen (50%)",
                "long": "🟩🟩🟩⬜ Long (75%)"
            }
            st.markdown(f"**{length_display[summary_length]}**")
            
            # Options pour le résumé
            structured_summary = st.toggle("Résumé structuré", value=True)
            
            # Language detection option
            st.subheader("Options linguistiques")
            auto_detect = st.toggle("Détection automatique de langue", value=True)
            
            if not auto_detect:
                lang_options = {
                    "fr": "🇫🇷 Français", 
                    "en": "🇬🇧 Anglais",
                    "ar": "🇸🇦 Arabe",
                    "es": "🇪🇸 Espagnol",
                    "de": "🇩🇪 Allemand"
                }
                lang = st.selectbox(
                    "Langue du contenu :",
                    options=list(lang_options.keys()),
                    format_func=lambda x: lang_options[x]
                )
            else:
                lang = None
                
            # Afficher options spécifiques à l'audio si mode audio sélectionné
            if option == "🎤 Audio":
                st.subheader("Options audio")
                task_type = st.radio(
                    "Type de tâche :",
                    ["transcribe", "translate"],
                    format_func=lambda x: "Transcrire" if x == "transcribe" else "Traduire en anglais",
                    horizontal=True
                )
                show_timestamps = st.toggle("Afficher les horodatages", value=False)
                
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Colonne de saisie et résultats à droite
    with col_input:
        tabs = st.tabs(["📝 Entrée", "📊 Résultat"])
        
        # Onglet d'entrée
        with tabs[0]:
            if option == "✏️ Texte":
                text_input = st.text_area(
                    "Collez votre texte ici :",
                    height=300,
                    placeholder="Entrez ou collez le texte à résumer ici...",
                    key="text_input"
                )
                
                if st.button("🔍 Générer le résumé", key="text_button", type="primary", use_container_width=True):
                    if not text_input.strip():
                        st.warning("⚠️ Veuillez entrer du texte avant de générer un résumé.")
                    else:
                        process_text(text_input, summary_length, auto_detect, lang, tabs, structured_summary)
            
            elif option == "📄 PDF":  # Option PDF
                uploaded_file = st.file_uploader(
                    "Téléversez un fichier PDF :",
                    type=["pdf"],
                    help="Le fichier doit être au format PDF et contenir du texte extractible."
                )
                
                # Aperçu PDF amélioré
                if uploaded_file:
                    display_pdf_preview(uploaded_file)
                    
                    if st.button("🔍 Générer le résumé", key="pdf_button", type="primary", use_container_width=True):
                        process_pdf(uploaded_file, summary_length, auto_detect, lang, tabs, structured_summary)
                else:
                    # Zone de glisser-déposer améliorée
                    st.markdown("""
                    <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 3rem; text-align: center;">
                        <h3>Glissez et déposez votre fichier PDF ici</h3>
                        <p>ou utilisez le bouton ci-dessus pour sélectionner un fichier</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            else:  # Option Audio
                uploaded_audio = st.file_uploader(
                    "Téléversez un fichier audio :",
                    type=["mp3", "wav", "m4a", "flac", "ogg", "aac", "wma"],
                    help="Le fichier doit être dans un format audio courant."
                )
                
                # Afficher un aperçu audio si un fichier est téléchargé
                if uploaded_audio:
                    display_audio_preview(uploaded_audio)
                    
                    # Afficher les détails du fichier audio
                    audio_info = {
                        "filename": uploaded_audio.name,
                        "size": round(len(uploaded_audio.getvalue())/1024, 1)
                    }
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📊 Fichier audio: {audio_info['size']} Ko")
                    
                    # Bouton de traitement
                    if st.button("🔍 Transcrire et résumer", key="audio_button", type="primary", use_container_width=True):
                        task = 'transcribe' if task_type == 'transcribe' else 'translate'
                        process_audio(uploaded_audio, summary_length, auto_detect, lang, tabs, structured_summary, task, show_timestamps)
                else:
                    # Zone de glisser-déposer pour l'audio
                    st.markdown("""
                    <div style="border: 2px dashed #ccc; border-radius: 10px; padding: 3rem; text-align: center;">
                        <h3>Glissez et déposez votre fichier audio ici</h3>
                        <p>Formats supportés: MP3, WAV, FLAC, OGG, M4A, AAC, WMA</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Footer amélioré
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.markdown("### 📱 Fonctionnalités")
        st.markdown("• Résumés intelligents\n• Support multilingue\n• Traitement de PDF & Audio")
    with col2:
        st.markdown("### 🛠️ Technologies")
        st.markdown("• Streamlit\n• NLP avancé\n• Traitement audio avec Whisper")
    with col3:
        st.markdown("### 💼 À propos")
        st.markdown("© 2025 Résumé Intelligent | Propulsé par IA")

def process_text(text_input, summary_length, auto_detect, lang, tabs, structured=True):
    """Traite le texte saisi et affiche le résultat dans l'onglet résultat."""
    with st.spinner("Analyse en cours..."):
        try:
            # Détection de langue
            detected_lang = detect(text_input) if auto_detect else lang
            
            # Génération du résumé (adapté pour le résumé structuré)
            if structured:
                structured_data = summarizer.summarize_structured(
                    text_input, 
                    summary_length,
                    detected_lang
                )
                summary = summarizer.format_structured_summary(structured_data, detected_lang)
            else:
                summary = summarizer.summarize_text(
                    text_input, 
                    summary_length,
                    detected_lang
                )
            
            # Affichage des résultats dans l'onglet résultat
            tabs[1].markdown("### ✅ Résumé généré avec succès")
            display_results(summary, detected_lang, tabs[1], structured_data if structured else None)
            
        except Exception as e:
            tabs[1].error(f"⚠️ Erreur lors du traitement : {str(e)}")

def process_pdf(uploaded_file, summary_length, auto_detect, lang, tabs, structured=True):
    """Traite le fichier PDF téléchargé et affiche le résultat dans l'onglet résultat."""
    with st.spinner("Traitement du PDF en cours..."):
        try:
            # Reset du curseur du fichier
            uploaded_file.seek(0)
            
            # Vérification de la validité du PDF
            if pdf_processor.validate_pdf(uploaded_file):
                # Reset du curseur après validation
                uploaded_file.seek(0)
                
                # Utilisation de la nouvelle méthode process_uploaded_file avec résumé structuré
                result = pdf_processor.process_uploaded_file(
                    uploaded_file, 
                    summary_length=summary_length,
                    lang=None if auto_detect else lang,
                    structured=structured
                )
                
                # Extraire les données importantes du résultat
                summary = result["summary"]
                structured_data = result.get("structured_data", None)
                pages_processed = result["pages_processed"]
                detected_lang = result["language"]
                title = result["title"]
                
                # Affichage des résultats dans l'onglet résultat
                tabs[1].markdown(f"### ✅ PDF traité avec succès ({pages_processed} page(s))")
                
                # Si un titre a été détecté, l'afficher
                if title:
                    tabs[1].markdown(f"**Titre détecté:** {title}")
                    
                display_results(summary, detected_lang, tabs[1], structured_data)
            else:
                # En cas d'échec de validation, essayer une méthode alternative
                try:
                    # Créer un fichier temporaire pour améliorer la compatibilité
                    uploaded_file.seek(0)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Extraction du texte et des sections avec la nouvelle méthode
                    try:
                        full_text, text_structure, total_pages = pdf_processor.extract_text_with_sections(
                            tmp_path, 
                            max_pages=10
                        )
                        
                        # Vérification minimale du contenu textuel
                        if len(full_text.strip()) < 20:
                            tabs[1].warning("⚠️ Le PDF contient très peu de texte extractible. Le résumé pourrait ne pas être pertinent.")
                        
                        # Détection de la langue si nécessaire
                        detected_lang = detect(full_text[:1000]) if auto_detect else lang
                        
                        # Générer le résumé (adapté pour le résumé structuré)
                        if structured:
                            structured_data = summarizer.summarize_structured(
                                full_text, 
                                summary_length,
                                detected_lang
                            )
                            summary = summarizer.format_structured_summary(structured_data, detected_lang)
                        else:
                            summary = summarizer.summarize_text(
                                full_text, 
                                summary_length,
                                detected_lang
                            )
                        
                        # Affichage des résultats
                        tabs[1].markdown(f"### ✅ PDF traité avec succès ({total_pages} page(s))")
                        
                        # Afficher le titre détecté s'il existe
                        if text_structure["title"]:
                            tabs[1].markdown(f"**Titre détecté:** {text_structure['title']}")
                            
                        display_results(summary, detected_lang, tabs[1], structured_data if structured else None)
                    
                    finally:
                        # Nettoyer le fichier temporaire
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                except Exception as alt_e:
                    tabs[1].error("⚠️ Le fichier PDF est invalide ou ne contient pas de texte extractible.")
                    tabs[1].info("""
                    💡 Conseils:
                    - Vérifiez que votre PDF n'est pas une image scannée (nécessite OCR)
                    - Vérifiez que le PDF n'est pas protégé ou endommagé
                    - Essayez un autre fichier PDF contenant du texte sélectionnable
                    """)
                    st.expander("Détails techniques").write(f"Erreur: {str(alt_e)}")
        
        except Exception as e:
            tabs[1].error(f"⚠️ Erreur lors du traitement : {str(e)}")
            tabs[1].info("💡 Conseil: Vérifiez le format de votre fichier PDF et réessayez.")

def process_audio(uploaded_audio, summary_length, auto_detect, lang, tabs, structured=True, task='transcribe', show_timestamps=False):
    """Traite le fichier audio téléchargé et affiche le résultat dans l'onglet résultat."""
    with st.spinner("Traitement de l'audio en cours..."):
        try:
            # Créer un fichier temporaire pour l'audio
            uploaded_audio.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_audio.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_audio.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Transcription de l'audio
                tabs[1].info("🔊 Transcription audio en cours... Cela peut prendre quelques instants.")
                
                if show_timestamps:
                    # Transcription avec horodatages
                    result = audio_processor.transcribe_with_timestamps(
                        tmp_path,
                        language=None if auto_detect else lang
                    )
                    transcript = result["text"]
                    
                    # Afficher la transcription avec horodatages en expander
                    with tabs[1].expander("Transcription avec horodatages"):
                        for segment in result["segments"]:
                            start = segment["start"]
                            end = segment["end"]
                            text = segment["text"]
                            st.markdown(f"**[{start:.2f}s - {end:.2f}s]** {text}")
                else:
                    # Transcription simple
                    transcript = audio_processor.transcribe_to_text(
                        tmp_path,
                        language=None if auto_detect else lang
                    )
                
                # Récupérer la langue de la transcription
                detected_lang = detect(transcript[:1000]) if auto_detect else lang
                
                # Génération du résumé
                tabs[1].info("📝 Génération du résumé en cours...")
                
                if structured:
                    structured_data = summarizer.summarize_structured(
                        transcript, 
                        summary_length,
                        detected_lang
                    )
                    summary = summarizer.format_structured_summary(structured_data, detected_lang)
                else:
                    summary = summarizer.summarize_text(
                        transcript, 
                        summary_length,
                        detected_lang
                    )
                
                # Affichage des résultats
                tabs[1].markdown("### ✅ Audio traité avec succès")
                
                # Afficher la transcription complète dans un expander s'il n'y a pas d'horodatages
                if not show_timestamps:
                    with tabs[1].expander("Transcription complète"):
                        st.markdown(transcript)
                
                # Afficher le résumé
                display_results(summary, detected_lang, tabs[1], structured_data if structured else None)
                
            finally:
                # Nettoyer le fichier temporaire
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        except Exception as e:
            tabs[1].error(f"⚠️ Erreur lors du traitement audio : {str(e)}")
            tabs[1].info("""
            💡 Conseils:
            - Vérifiez que votre fichier audio est dans un format valide
            - Vérifiez que le fichier audio contient des paroles clairement audibles
            - Essayez un fichier plus court pour le premier test
            """)

def display_pdf_preview(uploaded_file):
    """Affiche un aperçu amélioré du PDF téléchargé."""
    with st.expander("📄 Aperçu du PDF", expanded=True):
        # Création d'un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Tenter d'utiliser PyMuPDF pour l'aperçu
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(tmp_path)
                
                if doc.page_count > 0:
                    # Affichage de la première page
                    page = doc[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(0.6, 0.6))
                    img_bytes = pix.tobytes("png")
                    
                    # Informations sur le document
                    st.image(img_bytes, use_column_width=True)
                    
                    # Extraction d'un échantillon de texte pour vérification
                    sample_text = page.get_text()[:200] if len(page.get_text()) > 0 else "Aucun texte détecté"
                    
                    # Afficher les métadonnées du PDF
                    metadata = doc.metadata
                    title = metadata.get('title', 'Non disponible')
                    author = metadata.get('author', 'Non disponible')
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📊 Document: {doc.page_count} page(s) • {round(len(uploaded_file.getvalue())/1024, 1)} Ko")
                    with col2:
                        if title != 'Non disponible' or author != 'Non disponible':
                            st.info(f"📝 Métadonnées: {title} • {author}")
                    
                    # Aperçu du contenu textuel
                    if sample_text and sample_text != "Aucun texte détecté":
                        with st.expander("Aperçu du contenu textuel"):
                            st.text(sample_text + "...")
                    else:
                        st.warning("⚠️ Aucun texte facilement extractible détecté dans ce PDF.")
                else:
                    st.warning("⚠️ Le PDF semble être vide.")
                
                doc.close()
                
            except Exception as e:
                # Méthode alternative d'affichage si PyMuPDF échoue
                st.warning(f"⚠️ Aperçu limité: {str(e)}")
                st.info(f"📄 Fichier: {uploaded_file.name} • {round(len(uploaded_file.getvalue())/1024, 1)} Ko")
                
                # Afficher un cadre d'aperçu générique
                st.markdown("""
                <div style="border: 1px solid #ddd; padding: 20px; text-align: center; background-color: #f5f5f5; border-radius: 5px;">
                    <h3>📄 Fichier PDF chargé</h3>
                    <p>L'aperçu n'est pas disponible, mais le fichier a été chargé avec succès.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.warning(f"⚠️ Impossible d'afficher l'aperçu: {str(e)}")
        
        # Nettoyage du fichier temporaire
        try:
            os.unlink(tmp_path)
        except:
            pass

def display_audio_preview(uploaded_audio):
    """Affiche un aperçu du fichier audio téléchargé."""
    with st.expander("🔊 Aperçu audio", expanded=True):
        # Afficher le player audio directement
        st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")
        
        # Afficher les informations sur le fichier audio
        st.info(f"📊 Fichier: {uploaded_audio.name} • {round(len(uploaded_audio.getvalue())/1024, 1)} Ko")
        
        # Informations supplémentaires sur le traitement audio
        st.markdown("""
        <div style="border: 1px solid #ddd; padding: 15px; background-color: #f5f5f5; border-radius: 5px;">
            <h4>🎵 Traitement en cours</h4>
            <p>Le fichier audio sera transcrit puis résumé. Ce processus peut prendre quelques instants selon la durée du fichier.</p>
        </div>
        """, unsafe_allow_html=True)

def display_results(summary, lang, container, structured_data=None):
    """Affiche les résultats du résumé avec des options de téléchargement et de copie."""
    # Affichage de la langue avec drapeau
    lang_flags = {
        "fr": "🇫🇷", "en": "🇬🇧", "ar": "🇸🇦", 
        "es": "🇪🇸", "de": "🇩🇪", "it": "🇮🇹", 
        "pt": "🇵🇹", "nl": "🇳🇱", "ru": "🇷🇺"
    }
    flag = lang_flags.get(lang, "🌐")
    
    # Carte stylisée pour le résumé
    formatted_summary = summary.replace('\n', '<br>')

    container.markdown(f"""
        <div style="border-radius: 10px; border: 1px solid #e0e0e0; padding: 1.5rem; margin-bottom: 1rem;">
            <div style="margin-bottom: 0.8rem;">
                <span style="background-color: #f0f2f6; padding: 5px 10px; border-radius: 15px; font-size: 0.9rem;">
                    {flag} Langue détectée: <code>{lang}</code>
                </span>
            </div>
            <h4>Résumé généré:</h4>
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;">
                {formatted_summary}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Si nous avons des données structurées, afficher quelques informations supplémentaires
    if structured_data:
        with container.expander("Détails du résumé structuré"):
            # Afficher le nombre de points clés
            key_points_count = len(structured_data.get("key_points", []))
            st.info(f"📊 **Structure détectée:** {key_points_count} points clés identifiés")
            
            # Afficher la répartition du contenu
            intro_len = len(structured_data.get("intro", "").split())
            conclusion_len = len(structured_data.get("conclusion", "").split())
            total_len = intro_len + conclusion_len + sum(len(kp.split()) for kp in structured_data.get("key_points", []))
            
            # Créer un petit graphique de répartition
            if total_len > 0:
                intro_percent = int((intro_len / total_len) * 100)
                conclusion_percent = int((conclusion_len / total_len) * 100)
                points_percent = 100 - intro_percent - conclusion_percent
                
                st.write("**Répartition du contenu:**")
                st.write(f"Introduction: {intro_percent}% • Points clés: {points_percent}% • Conclusion: {conclusion_percent}%")
    
    # Boutons d'action en ligne
    col1, col2 = container.columns(2)
    
    with col1:
        # Bouton de copie
        if container.button("📋 Copier le résumé", key="copy_button", use_container_width=True):
            st.toast("✅ Résumé copié dans le presse-papiers!")
    
    with col2:
        # Bouton de téléchargement
        container.download_button(
            label="⬇️ Télécharger le résumé",
            data=summary,
            file_name=f"resume_{lang}_{get_timestamp()}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Statistiques sur le résumé
    words = len(summary.split())
    chars = len(summary)
    
    container.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 0.7rem; border-radius: 5px; margin-top: 1rem;">
        <b>📊 Statistiques:</b> {words} mots • {chars} caractères
    </div>
    """, unsafe_allow_html=True)

def get_timestamp():
    """Génère un horodatage pour les noms de fichiers."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    main()