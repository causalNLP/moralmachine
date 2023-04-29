import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
tqdm.pandas()


"""   How to configure API key for Deepl and Google Translate
DEEPL: login and activate API key here: https://www.deepl.com/en/account/ then as usual set the environment variable
GOOGLE: Login in google cloud and follow this instructions: https://cloud.google.com/translate/docs/setup

"""
    
    
class MultiTranslator:
    def __init__(self):
        
        #import libraries and set environment variables only if needed
        from google.cloud import translate_v2 as translate  #needs the GOOGLE_APPLICATION_CREDENTIALS environment variable
        import deepl
        load_dotenv()
        DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
        
        
        self.deeplTranslator = deepl.Translator(DEEPL_API_KEY)
        self.google_translate = translate.Client()
        self.deepl_lang = ["it", "es", "fr", "de", "pt", "pl", "nl", "sv", "no", "fi", "da", "el", "cs", "hu", "bg", "ro", "en-us"]
        self.google_lang = ["ar", "zh", "ja", "ko", "hi", "ur", "bn", "ru", "tr", "sw", "id", "ms", "vi", "th", "he", "fa", "am", "uk"]
        
    def get_languages(self):
        return self.deepl_lang + self.google_lang
    
    def translate_text_deepl(self, text, source_lang, target_lang):
        return self.deeplTranslator.translate_text(text, source_lang = source_lang, target_lang= target_lang).text
    
    def translate_text_googleTranslate(self, text, source_lang, target_lang):
        return self.google_translate.translate(text, source_language=source_lang, target_language=target_lang)["translatedText"]
    
    def translate(self,text, source_lang, target_lang):
        if source_lang in ["en-us", "en-uk"]:
            if target_lang in self.deepl_lang:
                return self.translate_text_deepl(text, source_lang, target_lang)
            if target_lang in self.google_lang:
                return self.translate_text_googleTranslate(text, source_lang, target_lang)
            else:
                raise ValueError("Target language not supported")
        
        if target_lang in ["en-us", "en-uk"]:
            if source_lang in self.deepl_lang:
                return self.translate_text_deepl(text, source_lang, target_lang)
            if source_lang in self.google_lang:
                return self.translate_text_googleTranslate(text, source_lang, target_lang)
            else:
                raise ValueError("Source language not supported")
            
    def assert_supported_language(self, lang):
        if lang not in self.deepl_lang and lang not in self.google_lang:
            raise ValueError("Language not supported")
            
            
class TranslateResponse():
    
    def __init__(self):
        self.translator = MultiTranslator()
        self.response_folder_path = "data/cache"
        self.output_folder_path = "data/cache_translated"
        
    def iterate_over_folder(self):
        for filename in tqdm(os.listdir(self.response_folder_path), desc="Translating files"):
            if filename.endswith(".csv") and len(filename.split("_")) > 3:
                #get language from filename
                lang = filename.split("_")[2]
                if lang == "en" or len(lang) > 2:
                    continue   
                if lang not in self.translator.get_languages():
                    print("Language not supported: ", lang)   
                else:
                    self.translate_file(filename, lang)
    
    def translate_file(self, filename, lang):
        print("Translating file: ", filename, "from language: ", lang)
        temp_filename = filename.split(".")[0] + "_translated.csv"
        #check if file already exists
        if os.path.isfile(os.path.join(self.output_folder_path, temp_filename)):
            print("File already translated")
            return
        df = pd.read_csv(os.path.join(self.response_folder_path, filename))
        df["pred"] = df["pred"].progress_apply(lambda x: self.translator.translate(x, source_lang=lang, target_lang = "en-us"))
        df["query"] = df["query"].progress_apply(lambda x: self.translator.translate(x, source_lang=lang, target_lang = "en-us"))
        df.to_csv(os.path.join(self.output_folder_path, temp_filename ), index=False)
    
def main():
    translate_response = TranslateResponse()
    translate_response.iterate_over_folder()

if __name__ == '__main__':
    main()