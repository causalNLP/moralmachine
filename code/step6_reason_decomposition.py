from step8_compile_to_country_vec import ResponseReasonClassifier, LanguageFileManager


class ReasonFiles(LanguageFileManager):
    final_file_templ = 'data/control_{model_version}_{persona}_{lang}.csv'

    def _get_all_combinations(self, if_iterate_personas=True, if_iterate_langs=True,
                              if_iterate_model_versions=True, langs=None, model_versions=None):
        langs = self.load_lang_overview()['langs'] if langs is None else langs
        personas = ['default', 'expert']
        model_versions = ['gpt3', 'gpt3.5', 'gpt4'] if model_versions is None else model_versions

        combinations = []
        if if_iterate_model_versions:
            combinations += [[i, 'normal', 'en'] for i in model_versions]
        if if_iterate_personas:
            combinations += [['gpt4', i, 'en'] for i in personas]
            combinations += [['gpt3', i, 'en'] for i in personas]
        if if_iterate_langs:
            combinations += [['gpt3', 'normal', i] for i in langs]
        print(combinations)
        return combinations

    def __init__(self, combinations=None, **kwargs):
        super().__init__()

        if combinations is None:
            combinations = self._get_all_combinations(**kwargs)
        self.combinations = combinations

    def response2reason(self, openai_key_alias='OPENAI_API_KEY'):
        checker = ResponseReasonClassifier(openai_key_alias=openai_key_alias)
        import time
        time.sleep(10)

        from tqdm import tqdm
        for model_version, persona, lang in self.combinations:
            file = self.final_file_templ.format(model_version=model_version, persona=persona, lang=lang)
            from efficiency.log import fread
            data = fread(file)
            if not data:
                continue
            for row in tqdm(data, desc=file):
                response = row.get('gpt_response_en', row['gpt_response'])
                prompt = row.get('prompt_en_original', row['Prompt'])
                reasons = checker.check_response_quality(response, prompt)
                row['gpt_response_reason'] = reasons
            from efficiency.log import write_dict_to_csv
            write_dict_to_csv(data, file, verbose=True)

    def plot(self):
        file = 'data/reason_hierarchy.csv'
        from efficiency.log import fread
        df = fread(file, return_df=True)

        file_names = set(df.columns) - \
                     {'reason_type', 'description', 'moral_philosophy_branch_small',
                      'moral_philosophy_branch_big', 'moral_philosophy_branch_big_gpt4', 'percentage',
                      'improved_reason_type'}
        import plotly.express as px
        import plotly.io as pio

        for file_name in file_names:
            # model_version, system_role, lang = file_name.split('_', 2)
            custom_color_scale = ["#E0F3F3", "#C7EAEC", "#AEDDE6", "#95D0E0", "#7BC3DA", "#61B6D4", "#47A9CE"]
            # ["#f2d9e6", "#e3b5bc", "#d290a1", "#c26b87", "#b3456c", "#a41f52", "#95003a"]
            fig = px.sunburst(
                df, path=['moral_philosophy_branch_big', 'improved_reason_type'],
                values=file_name, color='moral_philosophy_branch_big',
                # color_continuous_scale=custom_color_scale,
                color_discrete_map={'Utilitarianism': '#E0B274', 'Fairness': '#8CC888', 'Others': '#9CBADE'},
            )


            fig.update_layout(font_size=30)
            # fig.show()
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),  # Adjust the values as needed
            )
            plot_file = f"data/fig/fig_reason_{file_name}.pdf"
            pio.write_image(fig, plot_file, format="pdf", width=600, height=600, scale=2)
            print(f'[Info] Generated figures in {plot_file}')

    def plot_get_raw_numbers(self):
        choices = ResponseReasonClassifier.choice2desc
        from efficiency.log import fread
        all_rows = []
        for model_version, persona, lang in self.combinations:
            file = self.final_file_templ.format(model_version=model_version, persona=persona, lang=lang)
            df = fread(file, return_df=True)
            if 'gpt_response_reason' not in df:
                print(f'[Warn] {file} does not have the "gpt_response_reason" column yet')
                continue
            reasons = df['gpt_response_reason'].str.split('; ').to_list()
            from collections import defaultdict
            reason_type2cnt = defaultdict(list)
            for reason_row in reasons:
                reason_row = {i if i in choices else "Others" for i in reason_row}
                if ('Others' in reason_row) and (len(reason_row) > 1):  # TODO: make all invalid strings others
                    reason_row.discard('Others')

                cnt = 1 / len(reason_row) / len(df)
                for reason_type in reason_row:
                    reason_type2cnt[reason_type].append(cnt)
            import numpy as np
            reason_type2cnt = {k: np.sum(v) for k, v in reason_type2cnt.items()}

            rows = [{"reason_type": k, "occurrences": v * 100,
                     "model_version": model_version, "system_role": persona, "lang": lang
                     }
                    for k, v in reason_type2cnt.items()]
            all_rows.extend(rows)

        import pandas as pd
        df = pd.DataFrame(all_rows)
        from efficiency.log import pivot_df
        dff = pivot_df(df, rows="reason_type", columns=["model_version", "system_role", "lang", ],
                       score_col="occurrences")
        import pdb;
        pdb.set_trace()
        dff.sort_values(["reason_type"], inplace=True)
        print(dff)
        # cnt_df.sort_values([1, 0], inplace=True, ascending=False)
        # cnt_df[1] *= 100
        # print(file)
        # print(cnt_df)
        # # print_df_value_count(df, ['gpt_response_reason'])
        # cnt_df[1].sum()


if __name__ == '__main__':
    from run_toy_examples import get_args

    args = get_args()
    rf = ReasonFiles(if_iterate_personas=args.differ_by_system_roles, if_iterate_langs=args.differ_by_lang,
                     if_iterate_model_versions=args.differ_by_model, langs=args.langs,
                     model_versions=args.model_versions)
    if not args.scoring_only:
        rf.response2reason(args.api)
    # rf.plot_get_raw_numbers()
    rf.plot()
