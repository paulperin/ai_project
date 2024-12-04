from transformers import MarianMTModel, MarianTokenizer

def translate_text_multilanguages(text, language_source, languages_targets):
    """
    translate a text in several target languages.
    
    :param text: text to translate.
    :param language_source: Source language code (e.g. “fr” for french).
    :param languages_targets: List of target language codes (e.g. [“en”, “es”, “de”]).
    :return: Dictionary containing translations for each target language.
    """

    translations = {}
    
    for language_target in languages_targets:
        #Build template name for specified languages
        modele_nom = f"Helsinki-NLP/opus-mt-{source_language}-{language_target}" 
        
        #Load tokenizer and template
        tokenizer = MarianTokenizer.from_pretrained(modele_nom)
        model = MarianMTModel.from_pretrained(modele_nom)
        
        #Prepare the text for the model
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        #Translate the text
        translation = model.generate(**tokens)
        
        #Decode the result
        text_translate = tokenizer.decode(translation[0], skip_special_tokens=True)
        
        #Add the translation to dictionnary
        translations[language_target] = text_translate
    
    return translations


print("Welcome to the AI Translator!")
while True:
    #User interface
    text_source = input("\nWrite your text to be translated: ")
    source_language = input("Enter the language in which your text is written (e.g., 'fr' for French): ")
    languages_targets = input("Enter the languages into which you wish to translate your text, separated by commas (e.g., 'en,es,de'): ").split(',')

    #Translations
    translations = translate_text_multilanguages(text_source, source_language.strip(), [language.strip() for language in languages_targets])

    #Show results
    print("\nHere's your text translated into your chosen language(s):")
    for language, text_translate in translations.items():
        print(f"- {language} : {text_translate}")
    
    #Ask if the user wants to redo a translation
    next_to_continue = input("\nDo you want to translate another text? (yes/no): ").strip().lower()
    if next_to_continue not in ['yes', 'y']:
        print("Goodbye! Thank you for using the AI Translator.")
        break
