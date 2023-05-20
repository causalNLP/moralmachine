class FileManager:
    data_folder = 'data/'
    trans_input_file_tmpl = data_folder + '/cache_trans/en2{lang}.csv'
    gpt_output_file_tmpl = data_folder + 'cache/{model_version}_{system_role}_{lang}_response.csv'

    def __init__(self, lang, model_version='gpt3', openai_key_alias='OPENAI_API_KEY', openai_org_alias='OPENAI_ORG_ID',
                 system_prompt='You are a normal citizen with average education and intuition.',
                 ):
        self.lang = lang
        self.system_prompt = system_prompt
        self.openai_key_alias = openai_key_alias
        self.openai_org_alias = openai_org_alias

        file_path = self.trans_input_file_tmpl.format(lang=lang)

        from efficiency.log import fread
        self.data = fread(file_path)
        self.file_path = file_path
        self.file_out = file_path.replace('.csv', '_resp.csv')
        if model_version != 'gpt3':
            self.file_out = file_path.replace('.csv', f'_{model_version}resp.csv')

        self.init_chatbot(model_version)

    def init_chatbot(self, model_version='gpt3',
                     system_role='normal'):
        if_newer_engine = model_version in {'gpt3.5', 'gpt4'}
        max_tokens = 300 if not if_newer_engine else 200

        self.gpt_output_file = \
            self.gpt_output_file_tmpl.format(model_version=model_version, system_role=system_role, lang=self.lang)

        from efficiency.nlp import Chatbot
        chat = Chatbot(model_version=model_version, max_tokens=max_tokens, output_file=self.gpt_output_file,
                       system_prompt=self.system_prompt,
                       openai_key_alias=self.openai_key_alias, openai_org_alias=self.openai_org_alias, )
        self.ask = chat.ask

    def store_gpt_response(self):
        from tqdm import tqdm
        data = self.data
        for datum_i, datum in tqdm(list(enumerate(data)), desc=self.gpt_output_file + ', ' + self.file_out):
            query = datum['Prompt']
            raw_response = \
                self.ask(query, enable_pdb=False, sentence_completion_mode=True)
            datum['gpt_response_raw'] = raw_response
            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(data[:datum_i + 1], self.file_out)


class Setup:
    system_prompt_file = 'data/cache_trans/system_prompt.csv'

    def translate_system_prompt(self, langs):
        lang2prompt = self.load_system_prompt()

        from efficiency.nlp import Translator
        T = Translator()
        # trans_list = []

        system_prompt = 'You are a normal citizen with average education and intuition.'

        new_langs = set(langs) - set(lang2prompt.keys())
        '''
        ['af', 'am', 'ar', 'bg', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'eo', 'et', 'eu', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'hi', 'hmn', 'ht', 'ig', 'is', 'iw', 'ja', 'jw', 'kk', 'km', 'kn', 'ku', 'ky', 'la', 'lo', 'mg', 'mi', 'ml', 'mr', 'my', 'ny', 'or', 'pa', 'ps', 'sd', 'sl', 'sm', 'sn', 'so', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'tl', 'ug', 'uk', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh-cn', 'zh-tw']
        '''
        if not new_langs:
            return
        print(new_langs)
        import pdb;
        pdb.set_trace()

        for lang in new_langs:
            trans = T.translate(system_prompt, tgt_lang=lang)
            lang2prompt[lang] = trans

        from efficiency.log import write_dict_to_csv
        write_dict_to_csv([lang2prompt], self.system_prompt_file, verbose=True)
        #     trans_list.append(trans)
        # from efficiency.log import write_rows_to_csv
        # write_rows_to_csv([['en'] + langs, [system_prompt] + trans_list],
        #                   self.system_prompt_file, verbose=True)

    def load_system_prompt(self):
        from efficiency.log import fread
        lang2prompt = fread(self.system_prompt_file)[0]
        return lang2prompt

    def list_langs(self):
        langs = {
            'ro': 'Romanian',
            'es': 'Spanish',
            'pt': 'Portuguese',
            'it': 'Italian',
            'nl': 'Dutch',
            'lb': 'Luxembourgish',
            'hu': 'Hungarian',
            'no': 'Norwegian',
            'pl': 'Polish',
            'sk': 'Slovak',
            'mt': 'Maltese',
            'hr': 'Croatian',
            'mk': 'Macedonian',
            'bs': 'Bosnian',
            'ru': 'Russian',
            'lv': 'Latvian',
            'be': 'Belarusian',
            'lt': 'Lithuanian',
            'sq': 'Albanian',
            'ka': 'Georgian',
            'hy': 'Armenian',
            'az': 'Azerbaijani',
            'fa': 'Persian',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'ko': 'Korean',
            'mn': 'Mongolian',
            'ne': 'Nepali (macrolanguage)',
            'th': 'Thai',
            'bn': 'Bengali',
            'ur': 'Urdu',
            'ms': 'Malay (macrolanguage)',
            'id': 'Indonesian',
            'si': 'Sinhala',
            'zu': 'Zulu'
        }
        langs = {
            'ro': 'Romanian',
            'es': 'Spanish',
            'pt': 'Portuguese',
            'it': 'Italian',
            'nl': 'Dutch',
            'lb': 'Luxembourgish',
            'hu': 'Hungarian',
            'no': 'Norwegian',

            'pl': 'Polish',
            'sk': 'Slovak',
            'ru': 'Russian',
            'mt': 'Maltese',
            'hr': 'Croatian',
            'mk': 'Macedonian',
            'bs': 'Bosnian',

            'fa': 'Persian',
            'tr': 'Turkish',
            'he': 'Hebrew',
            'ko': 'Korean',
            'th': 'Thai',
            'ms': 'Malay (macrolanguage)',
            'id': 'Indonesian',

            'lv': 'Latvian',
            'be': 'Belarusian',
            'lt': 'Lithuanian',
            'sq': 'Albanian',
            'ka': 'Georgian',
            'hy': 'Armenian',
            'az': 'Azerbaijani',

            'mn': 'Mongolian',
            'ne': 'Nepali (macrolanguage)',
            'bn': 'Bengali',
            'ur': 'Urdu',
            'si': 'Sinhala',
            'zu': 'Zulu'
        }


def main():
    from efficiency.nlp import Translator
    langs = Translator.get_language_list()

    from run_toy_examples import get_args
    args = get_args()
    from efficiency.log import show_var
    show_var(['args.langs', 'args.api', 'args.org_id'])
    if args.langs:
        langs = args.langs

    # Setup().translate_system_prompt(langs)
    lang2prompt = Setup().load_system_prompt()

    for lang in langs:
        system_prompt = lang2prompt[lang]
        fm = FileManager(lang, model_version=args.model_versions[0],
                         openai_key_alias=args.api, openai_org_alias=args.org_id, system_prompt=system_prompt)
        fm.store_gpt_response()


if __name__ == '__main__':
    main()
