"""
This script computes LLM completion for a given prompt using Meta-LLAMA-3-8B model. It has support for instruct models.
For instruct model we rely in the chat template and ask the model to respond in JSON format.
The input file is a csv file with a column named 'Prompt' containing the prompts.
"""
import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
import csv
import pandas as pd
import os
import json
os.environ["HF_TOKEN"] = ""


def parse_dict_from_json(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Handle the case where the input is not a valid JSON string
        return {}

def main():
    #model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto", load_in_8bit=True)
    
    outputFileName = './llama_response.csv'
    with open(outputFileName, 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        row=['prompt','pred_raw', 'pred','model']
        writer.writerow(row)
    df = pd.read_csv('./moral_no_dup.csv')
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    with torch.no_grad():
        for i in range(0,df.shape[0],1):
            prompt=list(df['Prompt'].values)[i]
            if 'instruct' in model_id.lower():
                prompt_complete= "You are a normal citizen with average education and intuition.\n"+prompt+"? \n Please respond in JSON format with the key 'response':"

                messages = [
                    {"role": "user", "content": prompt_complete},
                ]

                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
            else:
                prompt_complete= "You are a normal citizen with average education and intuition.\n"+prompt
                input_ids = tokenizer.encode(
                    prompt,return_tensors="pt").to(model.device)
            outputs = model.generate(
                input_ids,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=False,
            )
            response = outputs[0][input_ids.shape[-1]:]

            response_decoded =tokenizer.decode(response, skip_special_tokens=True)
            if 'instruct' in model_id.lower():
                response_text = parse_dict_from_json(response_decoded)["response"]
            else:
                response_text = response_decoded
            with open(outputFileName, 'a', newline='') as csvoutput:
                writer = csv.writer(csvoutput)
                writer.writerow([prompt_complete] + [response_decoded] + [response_text] + [model_id])

if __name__ == '__main__':
    main()