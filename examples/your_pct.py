import matplotlib.pyplot as plt

import sys
sys.path.append("../")

from questionnaire import LikertQuestionnaire, AdministerHuman
from questionnaire.utils.pct_viz import make_pct_frame, place_image, place_tick


def main():
    # 1. load questionnaire
    pct_questionnaire = LikertQuestionnaire.from_json(
        json_path = "../data/pct.json",
        data_key="data",
        **{
            "prompt_template":"You can only choose one option. Respond only with the label of your answer. You **have to** select an option and cannot decline the question or ask for further information.\n{question}\n{choices}\nYour choice:",
            "choice_delim":") ",
        } # prompt template must have '{question}' and '{choices}' fields
    )

    # 2. run questionnaire
    your_pct = AdministerHuman(pct_questionnaire)
    your_results = your_pct.run()

    fig, ax = make_pct_frame(size=8)
    place_tick(
        x_pos=your_results["economic"], y_pos=your_results["social"],
        ax=ax,
        **{"color":"red", "label":"Random Answers"}
    )
    ax.legend()
    plt.tight_layout()
    plt.show()

    return


if __name__ == '__main__':
    main()
