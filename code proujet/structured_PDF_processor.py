import fitz  # PyMuPDF
import re
import logging
import tempfile
import os
import sys
from typing import Tuple, Optional, Dict, Any, Union
from Structured_Text_Summarizer import StructuredTextSummarizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StructuredPDFProcessor:
    def __init__(self, summarizer=None):
        """
        Initialise le processeur PDF avec un résumeur structuré optionnel.
        
        Args:
            summarizer: Instance de StructuredTextSummarizer (créée si None)
        """
        if summarizer is None:
            try:
                # Import local pour éviter les problèmes de circular import
                from structured_text_summarizer import StructuredTextSummarizer
                self.summarizer = StructuredTextSummarizer()
            except ImportError:
                self.summarizer = StructuredTextSummarizer()
        else:
            self.summarizer = summarizer
            
        self.supported_formats = ['.pdf']
        
    def extract_text_with_sections(self, file_path: str, max_pages: int = 10) -> Tuple[str, Dict[str, Any], int]:
        """
        Extrait le texte d'un PDF avec identification des sections potentielles.
        
        Args:
            file_path: Chemin vers le fichier PDF
            max_pages: Nombre max de pages à traiter
            
        Returns:
            Tuple (texte complet, structure détectée, nombre de pages réelles)
        """
        doc = None
        try:
            doc = fitz.open(file_path)
            total_pages = min(len(doc), max_pages)
            
            # Pour stocker le texte par page et par section potentielle
            all_text = []
            text_structure = {
                "title": "",
                "sections": [],
                "headings": []
            }
            
            # Extraction du titre potentiel (de la première page)
            if total_pages > 0:
                try:
                    page = doc.load_page(0)
                    page_text = page.get_text("text")
                    
                    # Tentative d'identification du titre (première ligne non vide)
                    lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                    if lines:
                        potential_title = lines[0]
                        # Un titre ne devrait pas être trop long
                        if len(potential_title) <= 100 and len(potential_title.split()) <= 20:
                            text_structure["title"] = potential_title
                except Exception as e:
                    logger.warning(f"Erreur extraction titre: {str(e)}")
            
            # Extraction du texte page par page avec détection des sections
            current_section = ""
            section_content = []
            
            for page_num in range(total_pages):
                try:
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")
                    
                    # Nettoyage basique
                    page_text = page_text.strip()
                    if not page_text:
                        continue
                    
                    # Détection des titres de section potentiels
                    lines = page_text.split('\n')
                    for i, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Heuristique pour détecter un titre de section:
                        # - Ligne courte (moins de 60 caractères)
                        # - Peu de mots (moins de 10)
                        # - Souvent en majuscules ou avec une numérotation
                        # - Pas de ponctuation finale standard
                        if (len(line) < 60 and len(line.split()) < 10 and 
                            (line.isupper() or re.match(r'^[0-9IVX]+[.)]\s+\w+', line)) and
                            not line.strip().endswith(('.', ',', ';', ':', '?', '!'))):
                            
                            # Si on avait une section en cours, on la sauvegarde
                            if current_section and section_content:
                                text_structure["sections"].append({
                                    "heading": current_section,
                                    "content": '\n'.join(section_content)
                                })
                                text_structure["headings"].append(current_section)
                                section_content = []
                            
                            # Nouveau titre de section trouvé
                            current_section = line
                        else:
                            # Texte normal, l'ajouter à la section courante
                            # Nettoyage des espaces multiples
                            cleaned_line = re.sub(r'\s+', ' ', line).strip()
                            if cleaned_line:
                                section_content.append(cleaned_line)
                    
                    # Ajouter le texte complet de la page
                    cleaned_page_text = re.sub(r'\s+', ' ', page_text).strip()
                    all_text.append(cleaned_page_text)
                    
                except Exception as page_error:
                    logger.warning(f"Erreur page {page_num}: {str(page_error)}")
                    continue
            
            # Ne pas oublier la dernière section
            if current_section and section_content:
                text_structure["sections"].append({
                    "heading": current_section,
                    "content": '\n'.join(section_content)
                })
                text_structure["headings"].append(current_section)
            
            # Si aucune section n'a été détectée, traiter tout le texte comme une seule section
            if not text_structure["sections"] and all_text:
                text_structure["sections"].append({
                    "heading": "",
                    "content": ' '.join(all_text)
                })
            
            full_text = ' '.join(all_text)
            
            # Validation renforcée
            if len(full_text.split()) < 10:
                logger.warning("Peu de texte extrait - PDF peut être scanné")
                raise ValueError("Texte insuffisant - PDF peut être une image")
                
            return full_text, text_structure, total_pages
            
        except Exception as e:
            logger.error(f"Erreur extraction: {str(e)}")
            raise
        finally:
            if doc:
                doc.close()

    def process_uploaded_file(self, 
                             uploaded_file, 
                             summary_length="medium", 
                             lang=None, 
                             max_pages=10, 
                             structured=True) -> Dict[str, Any]:
        """
        Méthode spécialement pour Streamlit/Colab.
        Gère le fichier uploadé en mémoire sans écriture disque quand possible.
        
        Args:
            uploaded_file: Objet fichier uploadé (style Streamlit)
            summary_length: 'short', 'medium', 'long'
            lang: Code de langue (détection auto si None)
            max_pages: Nombre maximum de pages à traiter
            structured: Si True, génère un résumé structuré
            
        Returns:
            Dictionnaire avec le résumé et les métadonnées
        """
        try:
            # Sauvegarde du pointeur de fichier
            current_pos = uploaded_file.tell()
            uploaded_file.seek(0)  # Retour au début du fichier
            
            # Tentative d'utilisation directe du fichier en mémoire
            try:
                text = []
                text_structure = {
                    "title": "",
                    "sections": [],
                    "headings": []
                }
                
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                    total_pages = min(len(doc), max_pages)
                    
                    # Extraction du titre potentiel (de la première page)
                    if total_pages > 0:
                        try:
                            page = doc.load_page(0)
                            page_text = page.get_text("text")
                            
                            # Tentative d'identification du titre (première ligne non vide)
                            lines = [line.strip() for line in page_text.split('\n') if line.strip()]
                            if lines:
                                potential_title = lines[0]
                                # Un titre ne devrait pas être trop long
                                if len(potential_title) <= 100 and len(potential_title.split()) <= 20:
                                    text_structure["title"] = potential_title
                        except Exception as e:
                            logger.warning(f"Erreur extraction titre: {str(e)}")
                    
                    # Extraction du texte page par page avec détection des sections
                    current_section = ""
                    section_content = []
                    
                    for page_num in range(total_pages):
                        try:
                            page = doc.load_page(page_num)
                            page_text = page.get_text("text")
                            
                            # Nettoyage basique
                            page_text = page_text.strip()
                            if not page_text:
                                continue
                            
                            # Détection des titres de section potentiels
                            lines = page_text.split('\n')
                            for i, line in enumerate(lines):
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                # Heuristique pour détecter un titre de section:
                                if (len(line) < 60 and len(line.split()) < 10 and 
                                    (line.isupper() or re.match(r'^[0-9IVX]+[.)]\s+\w+', line)) and
                                    not line.strip().endswith(('.', ',', ';', ':', '?', '!'))):
                                    
                                    # Si on avait une section en cours, on la sauvegarde
                                    if current_section and section_content:
                                        text_structure["sections"].append({
                                            "heading": current_section,
                                            "content": '\n'.join(section_content)
                                        })
                                        text_structure["headings"].append(current_section)
                                        section_content = []
                                    
                                    # Nouveau titre de section trouvé
                                    current_section = line
                                else:
                                    # Texte normal, l'ajouter à la section courante
                                    # Nettoyage des espaces multiples
                                    cleaned_line = re.sub(r'\s+', ' ', line).strip()
                                    if cleaned_line:
                                        section_content.append(cleaned_line)
                            
                            # Ajouter le texte complet de la page
                            cleaned_page_text = re.sub(r'\s+', ' ', page_text).strip()
                            text.append(cleaned_page_text)
                            
                        except Exception as page_error:
                            logger.warning(f"Erreur page {page_num}: {str(page_error)}")
                            continue
                    
                    # Ne pas oublier la dernière section
                    if current_section and section_content:
                        text_structure["sections"].append({
                            "heading": current_section,
                            "content": '\n'.join(section_content)
                        })
                        text_structure["headings"].append(current_section)
                
                full_text = ' '.join(text)
                if not full_text.strip() or len(full_text.split()) < 10:
                    raise ValueError("Aucun texte extrait ou texte insuffisant - PDF peut être scanné")

                if structured:
                    structured_summary = self.summarizer.summarize_structured(full_text, summary_length, lang)
                    formatted_summary = self.summarizer.format_structured_summary(structured_summary, lang)
                    
                    # Enrichissement avec les informations de structure
                    result = {
                        "summary": formatted_summary,
                        "structured_data": structured_summary,
                        "pages_processed": total_pages,
                        "document_structure": text_structure,
                        "title": text_structure["title"],
                        "language": lang or self.summarizer.detect_language(full_text)
                    }
                else:
                    # Résumé standard
                    summary = self.summarizer.summarize_text(full_text, summary_length, lang)
                    result = {
                        "summary": summary,
                        "pages_processed": total_pages,
                        "document_structure": text_structure,
                        "title": text_structure["title"],
                        "language": lang or self.summarizer.detect_language(full_text)
                    }
                
                return result
                
            except Exception as direct_error:
                # Si l'approche directe échoue, on essaie avec un fichier temporaire
                logger.warning(f"Lecture directe échouée: {str(direct_error)}, tentative avec fichier temporaire")
                uploaded_file.seek(0)  # Réinitialiser le pointeur
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                try:
                    if structured:
                        full_text, text_structure, total_pages = self.extract_text_with_sections(tmp_path, max_pages)
                        structured_summary = self.summarizer.summarize_structured(full_text, summary_length, lang)
                        formatted_summary = self.summarizer.format_structured_summary(structured_summary, lang)
                        
                        result = {
                            "summary": formatted_summary,
                            "structured_data": structured_summary,
                            "pages_processed": total_pages,
                            "document_structure": text_structure,
                            "title": text_structure["title"],
                            "language": lang or self.summarizer.detect_language(full_text)
                        }
                    else:
                        full_text, text_structure, total_pages = self.extract_text_with_sections(tmp_path, max_pages)
                        summary = self.summarizer.summarize_text(full_text, summary_length, lang)
                        
                        result = {
                            "summary": summary,
                            "pages_processed": total_pages,
                            "document_structure": text_structure,
                            "title": text_structure["title"],
                            "language": lang or self.summarizer.detect_language(full_text)
                        }
                    
                    return result
                    
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        logger.warning("Impossible de supprimer le fichier temporaire")
                        
        except Exception as e:
            logger.error(f"Erreur traitement: {str(e)}")
            raise
        finally:
            # Restaure le pointeur de fichier
            try:
                uploaded_file.seek(current_pos)
            except:
                pass

    def summarize_pdf_structured(
        self,
        file_path: str,
        summary_length: str = "medium",
        lang: Optional[str] = None,
        max_pages: int = 10
    ) -> Dict[str, Any]:
        """
        Pipeline complet: Extraction + résumé structuré.
        
        Args:
            file_path: Chemin vers le PDF
            summary_length: 'short', 'medium', 'long'
            lang: 'fr', 'en', 'ar' (auto si None)
            max_pages: Limite de pages
            
        Returns:
            Dictionnaire avec le résumé structuré et les métadonnées
        """
        full_text, text_structure, pages_processed = self.extract_text_with_sections(file_path, max_pages)
        structured_summary = self.summarizer.summarize_structured(full_text, summary_length, lang)
        formatted_summary = self.summarizer.format_structured_summary(structured_summary, lang)
        
        detected_lang = lang or self.summarizer.detect_language(full_text)
        
        result = {
            "summary": formatted_summary,
            "structured_data": structured_summary,
            "pages_processed": pages_processed,
            "document_structure": text_structure,
            "title": text_structure["title"],
            "language": detected_lang
        }
        
        return result
        
    def summarize_pdf(
        self,
        file_path: str,
        summary_length: str = "medium",
        lang: Optional[str] = None,
        max_pages: int = 10,
        structured: bool = True
    ) -> Union[Tuple[str, int], Dict[str, Any]]:
        """
        Pipeline complet: Extraction + résumé.
        Version compatible avec l'original, supporte les deux types de résumés.
        
        Args:
            file_path: Chemin vers le PDF
            summary_length: 'short', 'medium', 'long'
            lang: 'fr', 'en', 'ar' (auto si None)
            max_pages: Limite de pages
            structured: Si True, retourne un résumé structuré
            
        Returns:
            Si structured=False: Tuple (résumé, pages traitées)
            Si structured=True: Dict avec le résumé structuré et métadonnées
        """
        if structured:
            return self.summarize_pdf_structured(file_path, summary_length, lang, max_pages)
        else:
            full_text, _, pages_processed = self.extract_text_with_sections(file_path, max_pages)
            summary = self.summarizer.summarize_text(full_text, summary_length, lang)
            return summary, pages_processed

    def validate_pdf(self, file_or_path):
        """
        Vérifie rapidement si le PDF est valide.
        Accepte soit un chemin fichier, soit un objet file-like ou des bytes.
        """
        try:
            if isinstance(file_or_path, str):
                # C'est un chemin de fichier
                with fitz.open(file_or_path) as doc:
                    return len(doc) > 0
            elif isinstance(file_or_path, bytes):
                # C'est des bytes bruts
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(file_or_path)
                    tmp_path = tmp_file.name
                
                try:
                    with fitz.open(tmp_path) as doc:
                        result = len(doc) > 0
                    return result
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            else:
                # C'est un objet file-like
                current_pos = file_or_path.tell()
                file_or_path.seek(0)
                
                try:
                    with fitz.open(stream=file_or_path.read(), filetype="pdf") as doc:
                        return len(doc) > 0
                finally:
                    file_or_path.seek(current_pos)
        except Exception as e:
            logger.error(f"Erreur de validation PDF: {str(e)}")
            return False


# Exemple d'utilisation
if __name__ == "__main__":
    # Création du processeur
    pdf_processor = StructuredPDFProcessor()
    
    # Test sur un fichier PDF
    pdf_path = "exemple.pdf"
    try:
        # Résumé structuré
        result = pdf_processor.summarize_pdf(pdf_path, "medium", structured=True)
        print(f"Titre détecté: {result['title']}")
        print(f"Langue détectée: {result['language']}")
        print(f"Pages traitées: {result['pages_processed']}")
        print("\nRésumé structuré:")
        print(result['summary'])
        
        # Afficher les sections détectées
        print("\nStructure du document:")
        for section in result['document_structure']['sections']:
            print(f"- {section['heading']}")
            
    except Exception as e:
        print(f"Erreur: {str(e)}")