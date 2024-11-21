from copy import deepcopy
from abc import abstractmethod

#from models.questionning_models import *

# ================================================
class AdministerQuestionnaire:
    def __init__(
        cls,
        questionnaire,
    ):
        cls.questionnaire = questionnaire
        cls.answers = None

    @abstractmethod
    def _get_answer_probs(
        cls,
        prompts
    ):
        raise NotImplementedError()

    def _set_answers(
        cls,
        answers
    ):
        cls.answers = answers
        return 
    
    def run(
        cls,
        **kwargs
    ):
        # 1. make prompts
        prompts = cls.questionnaire.make_prompts()
        # 2. get answers probabilities
        answers_probabilities = cls._get_answer_probs(prompts, **kwargs)
        cls._set_answers(answers_probabilities)
        # 3. eval 
        results = cls.questionnaire.evaluate(
            answers_probabilities
        )
        return results

class AdministerCustom(AdministerQuestionnaire):
    def __init__(
        self,
        questionnaire,
        generation_method,
        output_parser,
        generation_args={},
        parser_args={},
        store_answers:bool=True
    ):
        """
        generation_method 
            - takes as inputs:
                - prompt (str) 
                - (+ `generation_args`)
        output_parser
            - takes as inputs:
                - generation_method() output
                - choice keys
                - (+ `parser_args`)
            -> output
                - dict with keys: choice keys and values: probabilities
                    (eg. {"A": .2, "B": .7, "C": .1})
        """
        super().__init__(
            questionnaire
        )
        self.generation_method = generation_method
        self.generation_args = generation_args
        self.output_parser = output_parser
        self.parser_args = parser_args
        self.store_answers = store_answers
        if self.store_answers:
            self.generated_responses = []

    def _get_answer_probs(
        self,
        prompts,
        **kwargs,
    ):
        generated_responses = [
            self.generation_method(p,**self.generation_args) 
            for p in prompts
        ]
        if self.store_answers:
            self.generated_responses = generated_responses
        cks = self.questionnaire.get_choices_keys()
        parsed_responses = [
            self.output_parser(r,cks[i],**self.parser_args)
            for i, r in enumerate(generated_responses)
        ]

        return parsed_responses

class AdministerHuman(AdministerCustom):
    def __init__(
        self,
        questionnaire,
    ):
        super().__init__(
            questionnaire,
            generation_method=self._get_input,
            output_parser=self._parse_input,
        )

    def _get_input(
        self,
        prompt:str,
    ):
        print(prompt)
        answer = input()
        return answer

    def _parse_input(
        self,
        input:str,
        choice_keys:list,
    ):
        parsed_input = {
            k: int(input.lower()==k.lower())
            for k in choice_keys
        }
        return parsed_input
        
    
class AdministerModel(AdministerQuestionnaire):
    def __init__(
        self,
        model_path:str,
        questionnaire,
        load_args:dict = {},
        gen_args:dict = {},
        # TODO: soft / hard choices
        # TODO: from generated text answer ?
    ):
        super().__init__(
            questionnaire
        )
        self.model, self.tokenizer = load_model(
            model_path = model_path,
            return_tokenizer=True,
            **load_args,
        )
        self.gen_args = gen_args

    def _get_answer_probs(
        self,
        prompts,
        hard_scores:bool=False,
        **kwargs,
    ):
        # 2. query model
        gen_answers, id_last = query_model(
            self.model, self.tokenizer,
            prompts = prompts,
            model_args = self.gen_args,
        )
        # 3. parse answers
        # 3.1 get logits
        logits = get_logits(
            gen_answers, id_last
        )
        # 3.2 get choices' indices token ids
        choice_ids = [
            get_tokens_ids(
                self.tokenizer,
                choice_keys,
                prefixes = [], # TODO ?
                suffixes = [], # TODO ?
                check_decode = False,
            )
            for choice_keys in self.questionnaire.get_choices_keys()
        ]
        # 3. get distribution over these tokens
        probs = get_tokens_probs(
            logits, choice_ids, normalize = True,
        )
        if hard_scores:
            probs = [
                {
                    k:int(v==max(question_p.values()))
                    for k,v in question_p.items()
                }
                for question_p in probs
            ]
        return probs

