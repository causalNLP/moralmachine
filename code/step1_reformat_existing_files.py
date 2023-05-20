class ExistingFileReader:
    existing_file_pattern = 'data/control_gpt3_normal_{lang}.csv'
    out_file_pattern = 'data/cache_trans/en2{lang}{suffix}.csv'
    suffix2num_keys = {
        '': 4,
        '_resp': 5,
        '_resp_tr': 6,
    }
    keys = ["two_choices", "prompt_en_original", "prompt_en", "Prompt", "gpt_response_raw", "gpt_response_en", ]
    key_remap = {
        ("gpt_response_raw", "gpt_response"): ("gpt_response_raw", "gpt_response_en"),
        ("gpt_response"): ("gpt_response_raw"),
    }
    final_file_templ = 'data/control_{model_version}_{persona}_{lang}.csv'
    out_file_en = 'data/cache_trans/en.csv'

    def save_vignettes(self):
        from efficiency.log import fread
        file = self.final_file_templ.format(model_version='gpt3', persona='normal', lang='en')
        df = fread(file, return_df=True)
        df = df[['two_choices', 'Prompt']].drop_duplicates()
        df.to_csv(self.out_file_en, index=False)
        print(f'Saved {len(df)} samples to {self.out_file_en}')

    def sanity_check_vignettes(self):
        personas = ['default', 'normal', 'expert']
        model_versions = ['gpt3.043', 'gpt3.5', 'gpt4']
        langs = ['en', 'es', 'de', 'zh-cn', 'it', 'ru']
        files = []
        files += [self.final_file_templ.format(model_version='gpt3', persona='normal', lang=lang) for lang in langs]
        files += [self.final_file_templ.format(model_version=i, persona='normal', lang='en') for i in model_versions]
        from efficiency.log import fread, show_var
        prompts_all = []
        for file in files:
            df = fread(file, return_df=True)
            if not len(df):
                continue
            for key in ['prompt_en_original', 'prompt_en', 'Prompt']:
                try:
                    df = df[['two_choices', key]].drop_duplicates()
                    break
                except:
                    pass
            prompts = df[key].to_list()
            show_var(["len(prompts)", "file"])
            prompts_all.extend(prompts)
        prompts_all = set(prompts_all)
        show_var(["len(prompts_all)"])
        if len(prompts_all) != 460:
            import pdb;
            pdb.set_trace()

    def reformat_file_for(self, lang):
        file = self.existing_file_pattern.format(lang=lang)

        import os
        if not os.path.exists(file): return

        file_outs = [self.out_file_pattern.format(lang=lang, suffix=suffix)
                     for suffix, num_keys in self.suffix2num_keys.items()]
        # if all(os.path.exists(i) for i in file_outs): return

        from efficiency.log import fread
        df = fread(file, return_df=True, verbose=False)
        df['prompt_en_original'] = df['prompt_en']
        intersec = set(df.columns) & {"gpt_response_raw", "gpt_response"}

        if len(intersec) == 2:
            df.rename(columns={"gpt_response": "gpt_response_en"}, inplace=True)
        elif len(intersec) == 1:
            df.rename(columns={"gpt_response": "gpt_response_raw"}, inplace=True)

        from efficiency.log import show_var
        for suffix, num_keys in self.suffix2num_keys.items():
            file_out = self.out_file_pattern.format(lang=lang, suffix=suffix)
            if os.path.exists(file_out):
                data = fread(file_out)
                if len(data) == 460:
                    continue
            this_keys = self.keys[:num_keys]
            missing_keys = set(this_keys) - set(df.columns)
            show_var(['file_out', 'missing_keys'], joiner=', ')
            if not missing_keys:
                this_df = df[this_keys].drop_duplicates()
                print('New file has', len(this_df), 'rows')
                this_df.to_csv(file_out, index=False)


if __name__ == '__main__':
    file_reader = ExistingFileReader()
    file_reader.sanity_check_vignettes()
    file_reader.save_vignettes()
    import sys

    sys.exit()

    from efficiency.nlp import Translator

    langs = Translator.get_language_list()
    for lang in langs:
        file_reader.reformat_file_for(lang)
