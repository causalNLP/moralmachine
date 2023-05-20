class Plotter:
    def run_all(self):
        from glob import glob
        import os
        file_2a = 'data/control_gpt4_normal_en.csv'

        # self.files_2b = sorted(glob('data/control*_gpt4_normal_en.csv'))
        files_2b = ['data/controlchar_gpt4_normal_en.csv', 'data/controlchar_gpt3.5_normal_en.csv',
                    'data/controlchar_gpt3_normal_en.csv',
                    'data/control_gpt4_normal_en.csv', ]
        self.file_2b_pattern = 'data/controlchar_{model_version}_normal_en.csv'
        for file in files_2b:  # ideally, we want the balanced data in "controlchar", but for early results, we use "control"
            if os.path.isfile(file):
                print('[Info] Plotting for', file)
                self.get_fig2b(file)
                import pdb;pdb.set_trace()

        files_3a = ''
        self.get_fig2a(file)

    def compute_ACME(self, df, categories, groups, prefer_which=1, if_perc=True,
                     return_type=['dict', 'list_of_dicts', 'df'][-1]):
        '''
        Corr coefficient between the columns "phenomenon_category" and "this_saving_prob"
        '''
        from sklearn.linear_model import LinearRegression
        rows = []
        model = LinearRegression(fit_intercept=False)
        for category in categories:
            pref = groups[category][prefer_which]
            tmp = df[df['phenomenon_category'] == category]
            if len(tmp) == 0:
                print('[Warn] No data for', category)
                acme = 0
            else:
                X = tmp['this_group_name'] == pref
                X = X.astype(int)
                Y = tmp['this_saving_prob']
                acme = model.fit(X.values.reshape(-1, 1), Y).coef_[0]
            if if_perc: acme *= 100
            row = {'criterion': f'{category}_{pref}', 'acme': round(acme, 2)}
            rows.append(row)
        import pandas as pd
        df = pd.DataFrame(rows)
        df.sort_values(['criterion', 'acme'], inplace=True)
        if return_type == 'dict':
            dic = dict(zip(df['criterion'], df['acme']))
            return dic
        elif return_type == 'list_of_dicts':
            return rows
        elif return_type == 'df':
            return df

    def get_fig2a(self, file='data/control_gpt4_normal_en.csv', df=None, if_plot_in_python=False,
                  return_type=['dict', 'list_of_dicts', 'df'][-1], verbose=False):
        import pandas as pd
        import matplotlib.pyplot as plt

        categories = ['Gender', 'Fitness', 'SocialValue', 'Age', 'Utilitarianism', 'Species']
        groups = {
            "Species": ["Animals", "Humans"],
            "SocialValue": ["Low", "High"],
            "Gender": ["Male", "Female", ],
            "Age": ["Old", "Young", ],
            "Fitness": ["Unfit", "Fit", ],
            "Utilitarianism": ["Less", "More", ],
            # "Random": ["Rand", "Rand", ],
        }
        if df is None:
            data = pd.read_csv(file, index_col=None)
        else:
            data = df
        data['this_saving_prob'] = data['this_saving_prob'].replace(0.49, 0.5)  # this is the case of "to save either"
        data = data[data['this_saving_prob'] >= 0]
        data.index = range(len(data))

        df = self.compute_ACME(data, categories, groups, prefer_which=1,
                               return_type=return_type)
        # df2 = self.compute_ACME(data, categories, groups, prefer_which=0)
        # merged_df = pd.concat([df1, df2], axis=1)
        # import pdb;pdb.set_trace()
        if verbose:
            print(file)
            print(df)

        if if_plot_in_python:
            vals = df['acme']
            plt.barh(range(len(vals)), vals)
            plt.yticks(range(len(vals)),
                       [category + '\n' + "(" + groups[category][1] + " over " + groups[category][0] + ")" for category
                        in categories])

            # Save the figure to a PDF file with the specified DPI
            img_file = 'data/fig/fig_bar_pref_model.pdf'
            plt.savefig(img_file, dpi=300)
            print('[Info] Figure saved to', img_file)
        return df

    def get_fig2b(self, file, if_plot_in_python=False):
        '''
They compute it using randomly generated scenarios where on both sides there is only a single character.
Afterwards they compute Y = a_1*character_1 + a_2*character_2 + ... + a_n * character_n where a_i is the coefficient computed by the weighted linear regression,
and the character_i are all characters but Male and Female. In the plot they report then the a_i`s.
For now, I filter for every scenario from all the categories where there is only 1 character on both sides and
compute the lr as above (without weights).
        '''
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import LinearRegression

        # model = LogisticRegression(fit_intercept=False, penalty='none')
        model = LinearRegression()

        characters = ["Person", "Man", "Woman", "ElderlyMan", "ElderlyWoman", "Pregnant", "Stroller", "Boy", "Girl",
                      "Homeless", "LargeWoman", "LargeMan", "Criminal", "MaleExecutive", "FemaleExecutive",
                      "FemaleAthlete", "MaleAthlete", "FemaleDoctor", "MaleDoctor", "Dog", "Cat"]

        pos_class = ["Person", "Man", "Woman"]

        data = pd.read_csv(file, index_col=None)
        data = data[data['this_saving_prob'] >= 0]
        data.index = range(len(data))

        tmp = data[data['this_how_many_more_chars'] == 0].copy()

        # total number of chars in scence
        if 'Person' not in tmp.columns:
            characters.remove('Person')
            pos_class.remove('Person')
        chars_in_experiment = [c for c in characters if c in tmp.columns]
        tmp['number_of_chars'] = tmp[chars_in_experiment].sum(axis=1)

        tmp = tmp[tmp['number_of_chars'] == 1]

        X = tmp[chars_in_experiment].fillna(0).copy()
        # X['ManOrWomanOrPerson'] = X[pos_class].sum(axis=1)
        X.drop(columns=pos_class, inplace=True)

        Y = tmp['this_saving_prob']
        # print(X.astype(int))
        model.fit(X.astype(int), Y.astype(int))

        imp_scores = model.coef_
        chars = chars_in_experiment[len(pos_class):]
        # model_str = 'gpt4_preliminary_exp_estimates' if 'controlchar' not in file \
        #     else 'gpt3_estimates' if 'controlchar_gpt3' in file \
        #     else 'gpt4'
        pref, suff = self.file_2b_pattern.split('{model_version}', 1)
        from efficiency.function import lstrip_word, rstrip_word
        model_str = rstrip_word(lstrip_word(file, pref), suff)
        rows = [{'CharacterType': c, model_str: i} for c, i in zip(chars, imp_scores)]
        df = pd.DataFrame(rows)
        char2imp_score = dict(zip(chars, imp_scores))
        print(len(chars))
        print(len(imp_scores))
        df = df.sort_values(['CharacterType'])
        print(df)
        print(df[model_str].to_string(index=False))

        if not if_plot_in_python: return
        import pdb;
        pdb.set_trace()

        plt.barh(range(len(imp_scores)), imp_scores)
        plt.yticks(range(len(imp_scores)), chars)
        plt.xlim(-1, 1)
        plt.show()


def main():
    P = Plotter()
    P.run_all()
    # P.get_fig2a()


if __name__ == '__main__':
    main()
