# Moral Machine for LLMs

This repository contains the code for testing Moral Machine questions with LLMs.

### Code Structure

Our codes consist of the following elements:

1. `code/` folder

   - `question_generator.py`: Codes to generate the MoralMachine data consistent with the original paper
   - `run_toy_examples.py`: Codes to generate signature questions (so that the analysis can be super clean)
   - (Working) `run_toy_examples_multilingual.py ` and `multilingual_question_generator.py` .
   - `visualize_fig2.py`: codes to generate the original visualization of Figure 2a and 2b in the Moral Machine paper
   - `visualize_fig3.py`: codes to generate the original visualization of Figure 3a and 3b in the Moral Machine paper

2. `data/` folder

   - Input data for fig2 and fig3

   - Moral Machine responses by different LLMs in different languages

     

### Current Working Items

1. ZJ: working on (1) clustering the reasons given by the toy examples, and (2) check preferences over different roles.
2. FO: working on `run_toy_examples_multilingual.py ` and `multilingual_question_generator.py` .
3. AS: adding `get_fig2b()` to `question_generator.py`
4. FG: preparing to run hierarchical clustering in Fig3a and 3b once our output is ready.



