from questionnaire.administer_lab import AdministerCustom
from llm.inference_module import *
from llm.utils import *

class AdministerHF(AdministerCustom):
    def __init__(
        self,
        questionnaire,
        model_id:str,
        local:bool=False,
        logits_based:bool=False,
        load_args:dict={},
        generation_args:dict={},
        parser_args:dict={}
    ):
        super().__init__(
            questionnaire,
            generation_method=self._generation_method,
            output_parser=self._parse_answer,
            generation_args=generation_args,
        )
        if logits_based:
            assert local, "Not possible to retrieve logits from API. Set `logits_based` to False or run model locally."

        self.logits_based = logits_based
        self.local = local
        if self.local:
            self.inference_module = HFModelInference(
                model_id=model_id,
                load_args=load_args,
            )
        else:
            self.inference_module = HFSLAPIInference(
                model_id=model_id,
            )

    def _generation_method(
        self,
        input_str,
        **kwargs,
    ):
        return self.inference_module(
            input_str=input_str,
            **{
                "generation_args":self.generation_args,
                "logits_output":self.logits_based
            },
        )

    def _parse_answer(
        self,
        model_output,
        choices_keys:list,
        hard_scores:bool=False
    ):
        if self.logits_based:
            choice_ids = get_tokens_ids(
                self.inference_module.tokenizer,
                choices_keys,
                prefixes = [], # TODO ?
                suffixes = [], # TODO ?
                check_decode = False,
            )
            print(choice_ids)
            probs = get_tokens_prob(
                model_output, choice_ids, normalize = True,
            )
            if hard_scores:
                for k in probs.keys():
                    probs[k] = int(probs[k]==max(probs.values()))
        else:
            raise NotImplementedError
            
        return probs