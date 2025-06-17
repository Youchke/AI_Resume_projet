import pandas as pd
import numpy as np
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from evaluate import load
import nltk
from nltk.tokenize import sent_tokenize
import os

# Télécharger les ressources nltk nécessaires pour l'évaluation ROUGE
nltk.download('punkt')

# 1. Préparation des données
# Exemple de création de fichiers CSV (à adapter selon vos données)
def prepare_data_example():
    # Création d'un jeu de données plus complet pour l'entraînement
    train_data = {
        'text': [
            "Cet article présente les avancées récentes en intelligence artificielle. Plusieurs domaines sont abordés, notamment l'apprentissage profond, le traitement du langage naturel et la vision par ordinateur. Les chercheurs ont fait des progrès significatifs dans ces domaines. De nouvelles architectures de réseaux de neurones ont permis d'améliorer les performances dans diverses tâches. Les applications pratiques se multiplient dans des secteurs comme la santé, la finance et les transports.",
            
            "La France a connu une croissance économique modérée au premier trimestre 2023, avec une augmentation du PIB de 0,2%. Le secteur des services a été le principal moteur de cette croissance, tandis que l'industrie a continué à faire face à des défis. Le taux de chômage est resté stable à 7,1%, mais l'inflation a augmenté à 5,9% en glissement annuel, principalement en raison de la hausse des prix de l'énergie et de l'alimentation. Les économistes prévoient une légère amélioration pour le reste de l'année.",
            
            "Le changement climatique continue d'affecter les écosystèmes marins de la Méditerranée. Une étude récente a révélé que la température moyenne de la mer a augmenté de 1,2°C depuis les années 1980. Cette hausse de température provoque la migration de nombreuses espèces marines vers le nord et menace la biodiversité locale. Les chercheurs ont également observé une acidification croissante des eaux, ce qui affecte particulièrement les organismes à coquille calcaire. Des mesures de protection urgentes sont nécessaires pour préserver cet écosystème fragile.",
            
            "Le nouveau traitement contre le cancer du pancréas montre des résultats prometteurs lors des essais cliniques de phase II. Combinant immunothérapie et thérapie ciblée, ce protocole a permis de réduire significativement la taille des tumeurs chez 64% des patients traités. Les effets secondaires rapportés étaient généralement modérés et gérables. Les médecins espèrent que ce traitement pourra améliorer le pronostic de cette maladie particulièrement agressive. Une étude de phase III à plus grande échelle est prévue pour l'année prochaine.",
            
            "La réforme du système éducatif français, mise en œuvre depuis septembre dernier, fait l'objet de critiques de la part des syndicats d'enseignants. Les principales contestations concernent la réduction des effectifs dans certaines disciplines et l'introduction de nouvelles méthodes pédagogiques sans formation adéquate. Le ministère de l'Éducation nationale défend ces changements, arguant qu'ils visent à adapter l'école aux défis du 21e siècle et à améliorer les résultats des élèves français dans les classements internationaux. Une évaluation de l'impact de la réforme est prévue dans les prochains mois.",
            
            "Le festival international de cinéma de Cannes a attiré cette année plus de 200 000 visiteurs. La sélection officielle comprenait 24 films en compétition, représentant 15 pays différents. Le jury, présidé par la réalisatrice française Claire Denis, a décerné la Palme d'Or à un drame italien explorant les thèmes de l'immigration et de l'identité. Les critiques ont salué la diversité et la qualité artistique des œuvres présentées. Plusieurs producteurs ont profité de l'événement pour annoncer de nouveaux projets cinématographiques ambitieux.",
            
            "Une nouvelle technologie de batteries pour véhicules électriques promet d'augmenter l'autonomie de 40% tout en réduisant le temps de recharge à moins de 15 minutes. Développée par une startup française, cette batterie utilise un nouveau type d'électrolyte solide qui améliore la densité énergétique et la sécurité. Des tests sont en cours avec plusieurs constructeurs automobiles européens. Si les résultats sont concluants, la production de masse pourrait débuter dès 2024. Cette innovation pourrait contribuer significativement à l'adoption plus large des véhicules électriques.",
            
            "Le rapport annuel sur la cybersécurité révèle une augmentation de 35% des attaques de ransomware en France. Les secteurs les plus touchés sont la santé, l'éducation et les collectivités territoriales. Le coût moyen d'une attaque est estimé à 380 000 euros, sans compter les dommages à la réputation. Les experts recommandent aux organisations de renforcer leurs protocoles de sauvegarde, d'investir dans la formation du personnel et d'adopter une approche de sécurité à plusieurs niveaux. L'Agence nationale de la sécurité des systèmes d'information a publié de nouvelles directives pour aider les entreprises à se protéger.",
        ],
        'summary': [
            "Les avancées récentes en IA incluent des progrès en apprentissage profond, NLP et vision par ordinateur, avec de nouvelles architectures de réseaux neuronaux et des applications dans la santé, la finance et les transports.",
            
            "La France connaît une croissance économique modérée de 0,2% au premier trimestre 2023, portée par le secteur des services. Le taux de chômage reste stable à 7,1%, tandis que l'inflation atteint 5,9%, principalement due aux prix de l'énergie et de l'alimentation.",
            
            "Le changement climatique a augmenté la température de la Méditerranée de 1,2°C depuis les années 1980, provoquant la migration d'espèces vers le nord et une acidification des eaux qui menace la biodiversité marine, nécessitant des mesures de protection urgentes.",
            
            "Un nouveau traitement combinant immunothérapie et thérapie ciblée contre le cancer du pancréas montre des résultats prometteurs en phase II, réduisant la taille des tumeurs chez 64% des patients avec des effets secondaires modérés. Une étude de phase III est prévue.",
            
            "La réforme éducative française est critiquée par les syndicats d'enseignants pour la réduction des effectifs et l'introduction de nouvelles méthodes sans formation adéquate. Le ministère défend ces changements pour adapter l'école aux défis contemporains et améliorer les résultats dans les classements internationaux.",
            
            "Le Festival de Cannes a accueilli plus de 200 000 visiteurs avec 24 films en compétition de 15 pays. Le jury présidé par Claire Denis a décerné la Palme d'Or à un drame italien sur l'immigration et l'identité. L'événement a servi de plateforme pour annoncer de nouveaux projets cinématographiques.",
            
            "Une startup française développe une batterie pour véhicules électriques avec électrolyte solide augmentant l'autonomie de 40% et réduisant le temps de recharge à moins de 15 minutes. Des tests sont en cours avec des constructeurs européens pour une possible production en masse dès 2024.",
            
            "Les attaques par ransomware ont augmenté de 35% en France, ciblant principalement la santé, l'éducation et les collectivités territoriales, avec un coût moyen de 380 000 euros. Les experts recommandent des sauvegardes renforcées, la formation du personnel et une sécurité multiniveau.",
        ]
    }
    
    val_data = {
        'text': [
            "La situation économique en France montre des signes d'amélioration après la crise sanitaire. Les indicateurs économiques sont en hausse, notamment dans le secteur des services et du tourisme. Cependant, l'inflation reste une préoccupation majeure pour les ménages et les entreprises. La Banque de France prévoit une croissance de 1,2% pour l'année en cours, un chiffre revu à la baisse par rapport aux estimations précédentes.",
            
            "Une découverte archéologique majeure a été réalisée près de Lyon. Des chercheurs ont mis au jour les vestiges d'une villa romaine datant du IIe siècle après J.-C. Le site comprend des mosaïques exceptionnellement bien conservées, un système de chauffage par le sol et plusieurs statues en marbre. Cette découverte permet de mieux comprendre l'organisation sociale et la vie quotidienne à l'époque gallo-romaine. Le site sera ouvert au public après la fin des fouilles et la mise en place de mesures de conservation.",
            
            "La nouvelle politique agricole commune (PAC) de l'Union européenne suscite des réactions mitigées parmi les agriculteurs français. Si certaines mesures environnementales sont saluées, d'autres aspects concernant la répartition des aides sont critiqués. Les petites exploitations estiment que le système continue de favoriser l'agriculture intensive. Les syndicats agricoles appellent à des ajustements pour mieux soutenir l'agriculture familiale et les pratiques agroécologiques. Des négociations sont en cours au niveau national pour adapter le cadre européen aux spécificités françaises.",
            
            "Les résultats du baccalauréat 2023 montrent une légère baisse du taux de réussite par rapport à l'année précédente. 91,2% des candidats ont obtenu leur diplôme, contre 93,8% en 2022. Cette diminution s'explique en partie par le retour à des modalités d'examen plus classiques après les adaptations liées à la pandémie. Les filières technologiques enregistrent les baisses les plus significatives. Le ministère de l'Éducation nationale souligne néanmoins que les résultats restent supérieurs à ceux d'avant la crise sanitaire."
        ],
        'summary': [
            "L'économie française montre des signes d'amélioration après la crise sanitaire, particulièrement dans les services et le tourisme, mais l'inflation reste préoccupante. La Banque de France prévoit une croissance revue à la baisse de 1,2% pour l'année.",
            
            "Une villa romaine du IIe siècle a été découverte près de Lyon, comprenant des mosaïques bien conservées, un système de chauffage par le sol et des statues en marbre. Ce site archéologique majeur sera ouvert au public après les fouilles et la mise en place de mesures de conservation.",
            
            "La nouvelle PAC européenne reçoit des réactions mitigées des agriculteurs français, qui saluent certaines mesures environnementales mais critiquent la répartition des aides favorisant l'agriculture intensive. Les syndicats demandent des ajustements pour soutenir l'agriculture familiale et l'agroécologie.",
            
            "Le taux de réussite au baccalauréat 2023 a légèrement baissé à 91,2% contre 93,8% en 2022, principalement dans les filières technologiques, due au retour des modalités d'examen classiques après la pandémie. Les résultats restent toutefois supérieurs à ceux d'avant la crise sanitaire."
        ]
    }
    
    # Création des DataFrame pandas
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    # Sauvegarde en CSV
    train_df.to_csv('train_fr.csv', index=False)
    val_df.to_csv('val_fr.csv', index=False)
    
    print(f"Fichier train_fr.csv créé avec {len(train_df)} exemples")
    print(f"Fichier val_fr.csv créé avec {len(val_df)} exemples")

# 2. Chargement des données
def load_data():
    # Si vous avez déjà des fichiers CSV
    dataset = load_dataset('csv', data_files={
        'train': 'train_fr.csv',
        'validation': 'val_fr.csv'
    })
    
    # Alternative: créer un dataset à partir de DataFrames
    # train_df = pd.read_csv('train_fr.csv')
    # val_df = pd.read_csv('val_fr.csv')
    # dataset = {
    #     'train': Dataset.from_pandas(train_df),
    #     'validation': Dataset.from_pandas(val_df)
    # }
    
    return dataset

# 3. Tokenisation des textes
def preprocess_data(dataset, tokenizer, max_input_length=512, max_target_length=128):
    def preprocess_function(examples):
        # Tokenize les textes d'entrée
        inputs = examples['text']
        targets = examples['summary']
        
        model_inputs = tokenizer(
            inputs, 
            max_length=max_input_length,
            padding='max_length',
            truncation=True
        )
        
        # Tokenize les résumés cibles
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=max_target_length,
                padding='max_length',
                truncation=True
            )
            
        model_inputs['labels'] = labels['input_ids']
        
        # Remplacer les tokens de padding par -100 pour ignorer ces tokens lors du calcul de la loss
        for i in range(len(model_inputs['labels'])):
            # Remplacer les tokens de padding par -100
            padding_mask = labels['attention_mask'][i] == 0
            model_inputs['labels'][i][padding_mask] = -100
            
        return model_inputs
    
    # Appliquer la fonction de prétraitement
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names  # Supprimer les colonnes originales
    )
    
    return tokenized_dataset

# Fonction pour calculer les métriques ROUGE
def compute_metrics(eval_pred):
    rouge = load("rouge")
    predictions, labels = eval_pred
    
    # Décoder les prédictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Remplacer -100 par l'ID du token de padding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Postprocessing: découper en phrases
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Calculer les scores ROUGE
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    # Extraire les scores moyens
    result = {key: value * 100 for key, value in result.items()}
    
    return result

# 4. Configuration de l'entraînement
def setup_training(tokenized_dataset, model_name="moussaKam/barthez-orangesum-abstract"):
    # Charger le modèle de base
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Définir les arguments d'entraînement
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),  # Activer fp16 si GPU disponible
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        report_to="tensorboard"
    )
    # Créer le collateur de données pour le padding dynamique
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Créer le Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    return trainer

# 5. Entraînement du modèle
def train_model(trainer):
    # Lancer l'entraînement
    trainer.train()
    
    # Évaluer le modèle
    eval_results = trainer.evaluate()
    print(f"Résultats d'évaluation: {eval_results}")  
    return trainer

# 6. Évaluation et ajustement
def evaluate_model(trainer, tokenized_dataset):
    # Évaluation détaillée
    results = trainer.evaluate(tokenized_dataset["validation"])
    print(f"Résultats d'évaluation finaux: {results}")
    
    # Optionnel: Tester sur quelques exemples
    test_examples = tokenized_dataset["validation"].select(range(3))
    model = trainer.model
    
    for example in test_examples:
        input_ids = torch.tensor([example["input_ids"]]).to(model.device)
        attention_mask = torch.tensor([example["attention_mask"]]).to(model.device)
        
        generated_tokens = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
        
        decoded_summary = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"Résumé généré: {decoded_summary}")

# 7. Sauvegarde du modèle
def save_model(trainer, output_dir="./finetuned-barthez-model"):
    # Sauvegarder le modèle et le tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modèle sauvegardé dans: {output_dir}")
    return output_dir

# Fonction pour exécuter le pipeline complet
def run_fine_tuning_pipeline():
    # Mettre en place les répertoires
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # 1. Préparer et créer explicitement les fichiers CSV
    print("Étape 1: Création des fichiers de données...")
    prepare_data_example()
    
    # 2. Charger les données
    print("Étape 2: Chargement des données...")
    dataset = load_data()
    print(f"Données chargées: {dataset}")
    
    # 3. Tokenisation
    print("Étape 3: Tokenisation des textes...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez-orangesum-abstract")
    tokenized_dataset = preprocess_data(dataset, tokenizer)
    print(f"Tokenisation terminée. Données tokenisées: {tokenized_dataset}")
    
    # 4. Configuration de l'entraînement
    print("Étape 4: Configuration de l'entraînement...")
    trainer = setup_training(tokenized_dataset)
    
    # 5. Entraînement
    print("Étape 5: Entraînement du modèle...")
    trainer = train_model(trainer)
    
    # 6. Évaluation
    print("Étape 6: Évaluation du modèle...")
    evaluate_model(trainer, tokenized_dataset)
    
    # 7. Sauvegarde
    print("Étape 7: Sauvegarde du modèle fine-tuné...")
    model_path = save_model(trainer)
    
    print(f"Pipeline de fine-tuning terminé! Modèle sauvegardé dans: {model_path}")
    print(f"Fichiers de données créés: train_fr.csv et val_fr.csv")
    
    return model_path

# Fonction pour utiliser le modèle après entraînement
def generate_summary(text, model_path="./finetuned-barthez-model"):
    # Charger le modèle et le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Mettre le modèle en mode évaluation
    model.eval()
    
    # Tokenizer le texte
    inputs = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    # Générer le résumé
    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    with torch.no_grad():
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    # Décoder le résultat
    summary = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return summary

# Exemple d'utilisation du pipeline complet

