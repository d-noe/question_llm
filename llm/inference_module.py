from abc import abstractmethod

class LLMInference:
    def __init__(
        cls,
        model_id,
    ):
        cls.model_id = model_id

    @abstractmethod
    def prompt_model(
        cls,
        input_str,
        kwargs,
    ):
        raise NotImplementedError

    def __call__(
        cls,
        input_str,
        **kwargs,
    ):
        return cls.prompt_model(input_str, **kwargs)

# ====================================

class HFModelInference(LLMInference):
    def __init__(
        self,
        model_id,
        load_args:dict={}
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        super().__init__(
            model_id
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_args)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def prompt_model(
        self,
        input_str,
        logits_output:bool=True,
        generation_args={},
        tokenizer_args={}
    ):
        import torch
        tokenized_prompt = self.tokenizer.encode(
            input_str,
            return_tensors='pt', 
            **tokenizer_args
        )
        with torch.no_grad():
            output = self.model.generate(
                tokenized_prompt,
                output_scores=True,
                return_dict_in_generate=True,
                **generation_args,
            )

        if logits_output:
            id_last_token = tokenized_prompt.shape[1]-1
            model_response = output["scores"][id_last_token]
        else:
            model_response = self.tokenizer.decode(output["sequences"][0])
        
        return model_response


class HFSLAPIInference(LLMInference):
    def __init__(
        self,
        model_id,
    ):
        from huggingface_hub import InferenceClient
        super().__init__(
            model_id
        )
        self.client = InferenceClient(model_id)

    def prompt_model(
        self,
        input_str,
        generation_args={},
    ):
        client_response = self.client.chat_completion(
            messages=[{"role":"user", "content":input_str}],
            **generation_args,
        )
        model_response = client_response["choices"][0]["message"]["content"]
        return model_response

