import json
from string import digits
from string import ascii_lowercase as alc
import numpy as np

# ================================================
SURVEY_TEMPLATE = """Mark how much you agree with the following statement: {question}
{choices}
Provide only the letter corresponding to your answer.
"""
SURVEY_TEMPLATE = """Question: {question}
{choices}
Answer: """
SURVEY_TEMPLATE = """{question}
{choices}"""
# ================================================
draft_survey = {
    "categories": ["Cat A", "Cat B", "Cat C"],
    "data": {
        "id_0" : {
            "question": "Choose one answer from the following.",
            "choices":
            [
                "Answer 1",
                "Answer 2",
                "Answer 3",
                "Answer 4",
            ],
            "scores": [ # distribution of choice(i) over categories
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]
        },
        "id_1": {
            "question": "Choose an answer from the following.",
            "choices":
            [
                "Answer A",
                "Answer B",
                "Answer C",
            ],
            "scores": [ # distribution of choice(i) over categories
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ]
        },
        "id_2": {
            "question": "In the alphabet, after the letter 'B', comes ",
            "choices":[
                "A",
                "B",
                "C",
                "D",
            ],
            "scores": [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 1, 0]
            ],
        },
    }
}

# ================================================
class Questionnaire:
    def __init__(
        self,
        categories:list,
        questions:list,
        choices:list,
        scores:list,
        
        index_type:str="alphabetical",
        choice_delim:str=". ",
        prompt_template:str=SURVEY_TEMPLATE,
        #seed:int=None,
        **kwargs
    ):
        self.categories = categories
        
        self.questions = questions
        self.choices = choices
        self.scores = scores

        self.index_type = index_type
        self.choice_delim = choice_delim
        self.prompt_template = prompt_template

        self._set_choices_index(
            self.index_type,
            inplace=True
        ) # sets -> self.indices & self.index_type

        self.categories_bias = [0]*len(self.categories)
        if "categories_bias" in kwargs.keys():
            self.categories_bias = kwargs["categories_bias"]
            kwargs.pop("categories_bias")

        seed = None
        if "seed" in kwargs.keys():
            seed = kwargs["seed"]
            kwargs.pop("seed")
        #seed = kwargs["seed"] if 'seed' in kwargs.keys() else None
        if not seed is None:
            self._shuffle_choices(seed=seed)

    @classmethod
    def from_json(
        self,
        json_path,
        data_key:str = "data",
        **kwargs,
    ):
        with open(json_path) as f:
            loaded_survey = json.load(f)
        f.close()
        return self.from_dict(
            questionnaire_dict=loaded_survey,
            data_key=data_key,
            **kwargs
        )

    @classmethod
    def from_dict(
        self,
        questionnaire_dict,
        data_key = "data", 
        **kwargs,
    ):
        """
        format:
            survey = {
                "categories" : [...] ,
                "data" : {
                    "q_id" : {
                        "question" : ... ,
                        "choices" : [...] ,
                        "scores" : [...] ,
                    }
                }
            }
        
        return self(
            categories = questionnaire_dict["categories"],
            questions = [
                v["question"] 
                for v in questionnaire_dict["data"].values()
            ],
            choices = [
                v["choices"] 
                for v in questionnaire_dict["data"].values()
            ],
            scores = [
                v["scores"] 
                for v in questionnaire_dict["data"].values()
            ],
            **kwargs
        )
        """
        global_info = {
            k: v for k, v in questionnaire_dict.items()
            if not k==data_key
        }
        question_keys = list(
            questionnaire_dict[data_key].values()
        )[0].keys()
        question_info = {}
        for k in question_keys:
            if k.endswith('s'):
                question_info[k] = [v[k] for v in questionnaire_dict[data_key].values()]
            else:
                question_info["{}s".format(k)] = [v[k] for v in questionnaire_dict[data_key].values()]
        
        return self(
            **{**global_info, **question_info, **kwargs}
        )

    # =========
    
    def __getitem__(
        self,
        index
    ):
        return (
            self.questions[index], 
            self.choices[index], 
            self.scores[index]
        )
    
    def __len__(self):
        return len(self.questions)

    # =========

    def _set_choices_index(
        self,
        index_type:str,
        inplace:bool=False,
    ):
        if index_type=='alphabetical_l':
            indices = alc
        elif index_type=='alphabetical_u' or index_type=='alphabetical':
            indices = alc.upper()
        elif index_type=='numerical':
            indices = digits[1:]
        else:
            return NotImplementedError

        if inplace:
            self.index_type = index_type
            self.indices = list(indices)
        else:
            return list(indices)

    def _shuffle_choices(
        self,
        seed:int = None,
        inplace:bool=True,
    ):
        if not seed is None:
            np.random.seed(seed)
        n_ks = [len(cs) for cs in self.choices]
        reorder_indices = [
            np.random.choice(
                np.arange(len(cs)), len(cs), replace=False
            ).astype(int) for cs in self.choices
        ]

        rdm_choices = [
            [
                question_choices[rdm_id]
                for rdm_id in rdm_indices
            ]
            for question_choices, rdm_indices in zip(
                self.choices,
                reorder_indices
            )
        ]
        rdm_scores = [
            [
                question_scores[rdm_id]
                for rdm_id in rdm_indices
            ]
            for question_scores, rdm_indices in zip(
                self.scores,
                reorder_indices
            )
        ]

        if inplace:
            self.choices = rdm_choices#np.array(self.choices)[reorder_indices]
            self.scores = rdm_scores#np.array(self.scores)[reorder_indices]
            return 
        else:
            #return self.__init__(
            return type(self)(
                categories = self.categories,
                questions = self.questions,
                choices = rdm_choices,
                scores = rdm_scores,
                index_type = self.index_type,
                choice_delim = self.choice_delim,
                seed = None
            )

    # =========
    def make_prompts(
        self,
        shuffle_choices:bool=False,
    ):
        """
        prompt_template: "...{question}...{choices}..."
        """
        choices_str = [
            "\n".join(q_cs) if self.index_type is None else
            "\n".join([
                "{}{}{}".format(
                    self.indices[i],
                    self.choice_delim,
                    c
                )
                for i, c in enumerate(q_cs)
            ]) 
            for q_cs in self.choices
        ]
        prompts = [
            self.prompt_template.format(
                question = q,
                choices = cs
            )
            for q, cs in zip(self.questions, choices_str)
        ]
        
        return prompts

    def get_choices_keys(
        self,
    ):
        return [
            self.indices[:len(cs)]
            for cs in self.choices
        ]

    def _scores_to_dict(
        self,
    ):
        scores_dicts = [
            {
                choice_key: {
                    c:choice_score[i]
                    for i, c in enumerate(self.categories)
                }
                for choice_score, choice_key in zip(question_scores, question_choices)  # choice scores
            }
            for question_scores, question_choices in zip(self.scores, self.get_choices_keys()) # question scores
        ]
        return scores_dicts

    def evaluate(
        self,
        answers_probs:list,
        normalize_res:bool=False,
    ):
        """
        - TODO handle bias term! 
        answers_probs : [
            {...}, {...}
        ]
        """
        scores_dicts = self._scores_to_dict()
        results = {
            k:self.categories_bias[i]
            for i, k in enumerate(self.categories)
        }

        for choice_probs, question_scores in zip(answers_probs, scores_dicts):
            for choice_key, choice_prob in choice_probs.items():
                # check > 0 ? 
                for k in self.categories:
                    results[k] += question_scores[
                        choice_key # select scores for the considered answser
                    ][
                        k # select score for the considered category 
                    ]*choice_prob # weight by prob

        if normalize_res:
            tot_ = np.sum(list(results.values()))
            results = {k:v/tot_ for k,v in results.items()}
        # ? 
        """
        results["unclear"] = np.sum(
            [
                np.all([p==0 for p in question_as]) 
                for question_as in answers_ids
            ]
        )
        """
        return results

    # ========================
    def get_categories_scores(
        self,
        categories:list=None,
    ):
        if categories is None:
            categories = self.categories
        elif type(categories)==str:
            assert np.all([c in self.categories for c in categories])
            categories = [categories]
            
        categories_ids = [
            np.argmax([q_cat==cat for q_cat in self.categories])
            for cat in categories
        ]
            
        categories_scores = {
            cat: [
                [
                    (s[cat_id], a)
                    for a, s in zip(answers, scores)
                ]
                for _, answers, scores in self
            ] for cat_id, cat in zip(categories_ids, categories)
        }
            
        return categories_scores

    def _get_scores_range(
        self
    ):
        flat_scores = [
            s 
            for q_scores in self.scores 
            for qa_scores in q_scores
            for s in qa_scores
        ]
        return np.min(flat_scores), np.max(flat_scores)

    def get_optim_answers(
        self,
        category:str,
        neg_examples:bool=False,
        thrs:float = 0,
    ):
        assert category in self.categories
        min_s, max_s = self._get_scores_range()
        pos_thres = max_s-thrs
        neg_thres = min_s+thrs

        cat_scores_answers = self.get_categories_scores([category])[category]
        optim_answers = [
            [
                sa[1]
                for sa in scores_answers
                if sa[0] >= pos_thres
            ]
            for scores_answers in cat_scores_answers
        ]

        if neg_examples:
            neg_optim_answers = [
                [
                    sa[1]
                    for sa in scores_answers
                    if sa[0] <= neg_thres
                ]
                for scores_answers in cat_scores_answers
            ]
            return optim_answers, neg_optim_answers
            
        return optim_answers

# ================================================

class LikertQuestionnaire(Questionnaire): # TODO
    def __init__(
        self,
        categories:list, # N_categories
        questions:list, # N_q
        choice_descs:list, # N_choices 
        scores:list, # ??
        instruction:str=None, # TODO ??
        **kwargs
    ):
        super().__init__(
            categories=categories,
            questions=questions,
            choices=[choice_descs]*len(questions),
            scores=scores, # ??

            index_type = "numerical",
            seed = None, # doesn't make sense in Likert scale
            **kwargs
        )

# ====================

class TFQuestionnaire(Questionnaire): # TODO
    def __init__(
        self,
        questions:list,
        answers:list,
        choices:list=None,
        answer_type:str=None, # TODO
        **kwargs
    ):
        if choices is None:
            choices = [["True", "False"]]*len(questions)
        if answer_type is None: # -> exact match
            tf_scores = [
                [
                    [1, 0] if int(c == q_a) else [0, 1]
                    for c in q_cs
                ]
                for q_a, q_cs in zip(answers, choices)
            ]
        else:
            raise NotImplementedError
            
        super().__init__(
            categories = ["Correct", "Incorrect"],
            questions = questions,
            choices = choices,
            scores = tf_scores,

            seed=None, # doesn't make sense ??
            **kwargs
        )
    