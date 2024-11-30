from transformers import MarianMTModel, MarianTokenizer

def traduire_texte_multilangues(texte, langue_source, langues_cibles):
    """
    Traduit un texte dans plusieurs langues cibles.
    
    :param texte: Texte à traduire.
    :param langue_source: Code de la langue source (ex : "fr" pour français).
    :param langues_cibles: Liste des codes de langues cibles (ex : ["en", "es", "de"]).
    :return: Dictionnaire contenant les traductions pour chaque langue cible.
    langues dispo : en,fr,es,de,da,ru,he,sw,zh
    """
    traductions = {}
    
    for langue_cible in langues_cibles:
        # Construire le nom du modèle pour les langues spécifiées
        modele_nom = f"Helsinki-NLP/opus-mt-{langue_source}-{langue_cible}"
        
        # Charger le tokenizer et le modèle
        tokenizer = MarianTokenizer.from_pretrained(modele_nom)
        model = MarianMTModel.from_pretrained(modele_nom)
        
        # Préparer le texte pour le modèle
        tokens = tokenizer(texte, return_tensors="pt", truncation=True, padding=True)
        
        # Traduire le texte
        traduction = model.generate(**tokens)
        
        # Décoder le résultat
        texte_traduit = tokenizer.decode(traduction[0], skip_special_tokens=True)
        
        # Ajouter la traduction au dictionnaire
        traductions[langue_cible] = texte_traduit
    
    return traductions


# Entrées utilisateur
print("Welcome to the AI traductor !")
texte_source = input("Write your text to be translated : ")
langue_source = input("Enter the language in which your text is written (e.g. 'fr' for French) : ")
langues_cibles = input("Enter the languages into which you wish to translate your text, separated by commas (e.g. 'en,es,de') : ").split(',')

# Effectuer les traductions
traductions = traduire_texte_multilangues(texte_source, langue_source.strip(), [langue.strip() for langue in langues_cibles])

# Afficher les résultats
print("\nHere's your text translated into your chosen language(s) :")
for langue, texte_traduit in traductions.items():
    print(f"- {langue} : {texte_traduit}")
