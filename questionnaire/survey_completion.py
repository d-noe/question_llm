import argparse

from questionnaire.questionnaire_classes import *
from questionnaire.administer_lab import *

from survey_convert import *

# ================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", 
        help="Path to the model. If None (default): Human test.",
        type=str, default=None
    )
    parser.add_argument(
        "-s", "--survey_path", 
        help="Path to the survey, must be `.json` format and comply to the defined format to be loaded.",
        type=str, default=None
    )
    parser.add_argument(
        "-d", "--device",
        help="Device to use (eg. 'cpu' or 'cuda')",
        type=str, default='cpu'
    )
    parser.add_argument(
        "-v", "--verbose", 
        help="increase output verbosity",
        action="store_true"
    )
    parser.add_argument(
        "-ss", "--shuffle_seed", 
        help="Seed to shuffle questions' order in questionnaire.",
        type=int, default=None,
    )
    parser.add_argument(
        "-la", "--load_args",
        help = "Arguments given to 'load_model'. Format: key1:value1, key2:value2, ...",
        type = lambda x: {k:v for k,v in (i.split(':') for i in x.split(','))},
        default={}
    )
    
    args = parser.parse_args()

    # 1. load questionnaire
    if args.survey_path is None: # dev/test -> to be removed
        questionnaire = Questionnaire.from_dict(
            questionnaire_dict = draft_survey,
            seed = args.shuffle_seed
        )
    else:
        questionnaire = LikertQuestionnaire.from_json(
            json_path = args.survey_path,
            seed = args.shuffle_seed,
        )
    # 2. prepare pipeline
    if args.model_path is None:
        admin_questionnaire = AdministerHuman(
            questionnaire = questionnaire
        )
    else:
        admin_questionnaire = AdministerModel(
            model_path = args.model_path,
            load_args = args.load_args,
            questionnaire = questionnaire
        )    
    # 3. run
    results = admin_questionnaire.run()

    # display results
    if args.verbose:
        for k, v in results.items():
            print("{} -> {}".format(k,v))
    
    # 4. save results ?
    