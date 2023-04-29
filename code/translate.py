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
        
        try:
        #import libraries and set environment variables only if needed
            from google.cloud import translate_v2 as translate  #needs the GOOGLE_APPLICATION_CREDENTIALS environment variable
            import deepl
        except:
            print("Installing libraries...")
            os.system("pip install google-cloud-translate==2.0.1")
            os.system("pip install deepl==1.0.1")
            from google.cloud import translate_v2 as translate
            import deepl
        load_dotenv()
        DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
        #check if environment variable is set
        if DEEPL_API_KEY is None:
            raise ValueError("Environment variable DEEPL_API_KEY not set")
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is None:
            raise ValueError("Environment variable GOOGLE_APPLICATION_CREDENTIALS not set")
            
        
        
        self.deeplTranslator = deepl.Translator(DEEPL_API_KEY)
        self.google_translate = translate.Client()
        self.deepl_lang = ["it", "es", "fr", "de", "pt", "pl", "nl", "sv", "no", "fi", "da", "el", "cs", "hu", "bg", "ro", "en-us"]
        self.google_lang = ['ja', 'mk', 'bn', 'sk', 'ms', 'vi', 'sl', 'ar', 'ur', 'is', 'et', 'am', 'hr', 'bs', 'af', 'ka', 'ko', 'tr', 'id', 'sr', 'lv', 'hi', 'sw', 'th', 'fa', 'ru', 'uk', 'mt', 'lt', 'he', 'zh']
        
    def get_languages(self):
        return self.deepl_lang + self.google_lang
    
    def translate_text_deepl(self, text, source_lang, target_lang):
        return self.deeplTranslator.translate_text(text, source_lang = source_lang, target_lang= target_lang).text
    
    def translate_text_googleTranslate(self, text, source_lang, target_lang):
        return self.google_translate.translate(text, source_language=source_lang, target_language=target_lang)["translatedText"]
    
    def translate(self,text, source_lang, target_lang):
        if source_lang in ["en-us", "en-uk", "en"]:
            source_lang = "en"
            if target_lang in self.deepl_lang:
                return self.translate_text_deepl(text, source_lang, target_lang)
            if target_lang in self.google_lang:
                return self.translate_text_googleTranslate(text, source_lang, target_lang)
            else:
                raise ValueError(f"Target language {target_lang} not supported. Add it to MultiTranslator class")
        
        if target_lang in ["en-us", "en-uk", "en"]:
            if source_lang in self.deepl_lang:
                return self.translate_text_deepl(text, source_lang, target_lang)
            if source_lang in self.google_lang:
                target_lang = "en"
                return self.translate_text_googleTranslate(text, source_lang, target_lang)
            else:
                raise ValueError(f"Source language {source_lang} not supported. Add it to MultiTranslator class")
            
    def assert_supported_language(self, lang):
        if lang not in self.deepl_lang and lang not in self.google_lang:
            raise ValueError("Language not supported")
            
            
class TranslateResponse():
    
    def __init__(self):
        self.translator = MultiTranslator()
        self.response_folder_path = "data/cache"
        self.output_folder_path = "data/cache_translated"
        self.vignette_folder_path = "data/"
        self.vignette_output_folder_path = "data/vignette_translated"
        
    def iterate_over_folder(self):
        for filename in tqdm(os.listdir(self.response_folder_path), desc="Translating files"):
            if filename.endswith(".csv") and len(filename.split("_")) > 3:
                #get language from filename
                lang = filename.split("_")[2]
                if lang == "en" or len(lang) > 2:
                    continue   
                if lang not in self.translator.get_languages():
                    print("Language not supported: {lang} . Add it to MultiTranslator class")   
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
    
    def translate_vignette(self):
        for filename in tqdm(os.listdir(self.vignette_folder_path), desc = "Translating vignette:"):
            if filename.endswith(".csv") and filename.startswith("vignette"):
                for name in filename.split(".csv")[0].split("_"):
                    if len(name) == 2:
                        lang = name
                        break
                if lang == "en" or len(lang) > 2:
                    continue
                if lang not in self.translator.get_languages():
                    print(f"Language not supported: {lang}. Add it to MultiTranslator class")
                else:
                    self.translate_vignette_dataframe(filename, lang)
    
    def translate_vignette_dataframe(self, filename, lang):
        print("Translating vignette: ", filename, "from language: ", lang)
        temp_filename = filename.split(".")[0] + "_translated.csv"
        if os.path.isfile(os.path.join(self.vignette_folder_path, temp_filename)):
            print("File already translated")
            return
        df = pd.read_csv(os.path.join(self.vignette_folder_path, filename))
        df["Prompt"] = df["Prompt"].progress_apply(lambda x: self.translator.translate(x, source_lang=lang, target_lang = "en-us"))                
        df["two_choices"] = df["two_choices"].progress_apply(lambda x: self.translator.translate(x, source_lang=lang, target_lang = "en-us"))
        df["two_choices_unordered_set"] = df["two_choices_unordered_set"].progress_apply(lambda x: self.translator.translate(x, source_lang=lang, target_lang = "en-us"))
        df["gpt_response"] = df["gpt_response"].progress_apply(lambda x: self.translator.translate(x, source_lang=lang, target_lang = "en-us"))
        df.to_csv(os.path.join(self.vignette_folder_path, temp_filename), index=False)    

def main():
    translate_response = TranslateResponse()
    translate_response.iterate_over_folder()
    translate_response.translate_vignette()

if __name__ == '__main__':
    main()