from libretranslatepy import LibreTranslateAPI

lt = LibreTranslateAPI("http://localhost:5000/")


def translate_to_eng(text, source='ru', target='en'): 
    translated_text = lt.translate(text, source, target)

    return translated_text
    

def translate_to_rus(text, source = 'en', target='ru'):

    translated_text = lt.translate(text, source, target)

    return translated_text
