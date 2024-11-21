# Questionning LLMs

This repository contains code for interacting with and evaluating Large Language Models (LLMs) using questionnaires. 
It is made to be general enough to encompass different kind of experimental settings (including diverse questionnaires and alternative probing mechanisms). 
Yet, it is primarily developed to run the Political Compass Test on LLMs through HuggingFace's Serverless Inference API.

1. [⚙️ Getting Started](#getting-started)
2. [📚 Classes](#classes)
   1. [Questionnaire](#questionnaire)
   2. [AdministerLab](#administerlab)
3. [🤖 LLM Inference Modules](#llm)
   1. [Local HF Model](#local-hf)
   2. [HF Serverless Inference](#api-hf)
   3. [Character.ai (unoficial) API](#cai-api)


<!-- omit in toc -->
#  <a class="anchor" id="getting-started"></a> ⚙️ Getting Started

- [ ] make `requiremnts.txt` (+minimal?)
- [ ] step-by-step guide
- [ ] make overarching script.py

<!-- omit in toc -->
## Installation

- [ ] Clone this repository
- [ ] (Opt. create a virtural environment)
- [ ] Install the required dependencies
```bash
pip install -r requirements.txt
```
- [ ] test

<!-- omit in toc -->
## Examples

You can find examples in the [`examples` folder](./examples/):
- [`pct.ipynb`](./examples/pct.ipynb) walks through administering PCT to a LLM via HuggingFace's Serverless Inference API.
- [`your_pct.py`](./examples/your_pct.py) allows you to take the PCT and get your score on the political compass.

<!-- omit in toc -->
#  <a class="anchor" id="classes"></a> 📚 Classes

<!-- omit in toc -->
## <a class="anchor" id="questionnaire"></a> `Questionnaire`

The `Questionnaire` class provides the foundation for building different types of questionnaires. 
It takes as input:

* `categories`:  A list of categories used to classify the responses. Those are the output categories of the questionnaire.
* `questions`: A list of questions to be presented in the questionnaire.
* `choices`: A list of lists, where each sub-list contains the answer choices for the corresponding question.
* `scores`:  A list of lists, where each sub-list contains the scores assigned to each answer choice for the corresponding question. These scores assign a weihgt to each answer choice to the `categories` and are used to evaluate the output of the questionnaire based on the provided answers.

**Key functionalities of the `Questionnaire` class:**

*  **`make_prompts()`**:  Generates prompts for each question in the questionnaire, combining the question text and the answer choices into a user-defined template. 
*  **`evaluate()`**: Evaluates the responses based on the provided `answers_probs` (probabilities assigned to each answer choice) and the defined scoring scheme. 

**Subclasses of `Questionnaire`:**

*   **`LikertQuestionnaire`**: Designed for creating questionnaires with Likert scale questions, enabling the assessment of agreement or disagreement with statements. It uses numerical indices for answer choices (e.g., "1. Strongly Disagree", "2. Disagree", etc.).
*   **`TFQuestionnaire`**:  Used to create questionnaires with true/false questions, allowing for straightforward binary evaluations.


**Instantiating `Questionnaire`s:**

While `Questionnaire`s can be instantiated in code by providing the class (or subclass) with the required arguments, it also comes with **`from_json`** method that allows to create a `Questionnaire` instance from a `.json` file, using:
```python
questionnaire = Questionnaire.from_json("<PATH_TO_JSON_QUESTIONNAIRE>")
```
The `.json` file must comply to a strict format which includes the `categories` of the questionnaire, as well as a `data` field which contains the questions, choices (if not `LikertQuestionnaire` or `TFQuestionnaire`) and the `scores` which is a list of list where the first level is associated with the different choices and the second level represent the associated score distribution over the questionnaire's categories.

<details><summary>See an example of `.json` questionnaire format</summary>

```json
{
    "categories": ["Cat A", "Cat B", "Cat C"],
    "data": {
        "id_0" : {
            "question": "Choose one answer from the following.",
            "choices":
            [
                "Answer 1 (Cat A)",
                "Answer 2 (Cat B)",
                "Answer 3 (Cat C)",
                "Answer 4 (Cat A)",
            ],
            "scores": [ # distribution of choice(i) over categories
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0]
            ]
        },
        "id_1": {
            ...   
        },
        ...
    }
}
```
</details>

<!-- omit in toc -->
## <a class="anchor" id="administerlab"></a> `AdministerQuestionnaire`

### AdministerQuestionnaire

The `AdministerQuestionnaire` class handles the process of presenting questionnaires to participants (either human or not). It serves as a parent class defining the general structure and functionalities for administering questionnaires, while its subclasses implement specific methods tailored for different participant types.

**Key functionalities of the `AdministerQuestionnaire` class:**

*   **`_get_answer_probs()`**: This is an abstract method that needs to be implemented by each subclass. Its purpose is to obtain the probabilities of each answer choice for each question in the questionnaire.
*   **`run()`**: This method orchestrates the entire process of administering the questionnaire. 
    1.  First, it calls the `make_prompts()` method of the associated `Questionnaire` object to generate prompts for each question.
    2.  Next, it utilizes the subclass-specific `_get_answer_probs()` method to obtain the answer probabilities for each question.
    3.  Finally, it invokes the `evaluate()` method of the `Questionnaire` object, passing in the collected answer probabilities. The `evaluate()` method then calculates the overall score or result based on the predefined scoring scheme of the questionnaire.

**Subclasses of `AdministerQuestionnaire`:**

*   **`AdministerHuman`**: Designed for administering questionnaires to human participants, typically for testing or debugging purposes. It interacts with humans to obtain their answers. The `_get_answer_probs` method in this subclass is implemented using `_get_inputs` and `_parse_inputs`. `_get_inputs` is responsible for displaying the prompts to the user and collecting their inputs. `_parse_inputs` then processes the raw input strings, converting them into a dictionary format where keys represent answer choices and values are either 1 (selected) or 0 (not selected).

*   **`AdministerCustom`**: This class is a general and modular, yet non-abstract, subclass that allows to define the different modules of the `_get_answer_probs` method. It is typically thought to enable the interaction with LLMs in different and customisable ways. Upon initialization, it requires the user to provide a `generation_method` function (responsible for generating responses using the LLM) and an `output_parser` function (responsible for extracting answer probabilities from the generated responses). The `_get_answer_probs` method in this subclass utilizes these user-defined functions to obtain the answer probabilities.

<!-- omit in toc -->
#  <a class="anchor" id="llm"></a> 🤖 LLM Inference Module

The overall implementation is meant to offer flexibility in choosing how to interact with LLMs.
A few specific modules are pre-implemented, providing support for various inference modalities.

<!-- omit in toc -->
## <a class="anchor" id="local-hf"></a> Local HF Model

This module allows users to leverage their **locally hosted Hugging Face models**. Users need to provide the path to their model and tokenizer, and specify generation arguments.
It can be particularly useful to probe models and access their inner representations or output distributions (which might not be accessible through standard API services).

<!-- omit in toc -->
## <a class="anchor" id="api-hf"></a> HF Serverless Inference

This module allows users to utilize **Hugging Face's serverless inference API**. This option offers scalability and convenience for users who prefer not to manage local infrastructure, or simply when not available.


<!-- omit in toc -->
## <a class="anchor" id="cai-api"></a> Character.ai (unoficial) API

This module offers an interface for interacting with **LLMs hosted on Character.ai**, a platform designed for **conversational and character-driven interactions**. This module uses an unofficial API, so users should be aware of potential limitations or changes.
