if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, help="Path to the checkpoint file")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = AutoModelForCausalLM.from_pretrained("data/mixtera")
    model = model.cuda()

    prompt = args.prompt

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    output = model.generate(input_ids, max_length=100)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output_text)