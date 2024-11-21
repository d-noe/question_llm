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
    ):
        return model_response

# ====================================

class HFModelInference(LLMInference):
    def __init__(
        cls,
        model_id,
    ):
        from transformers import AutoTokenizer, AutoModelForCausalLM


class HFSLAPIInference(LLMInference):
    def __init__(
        cls,
        model_id,
    ):
        from huggingface_hub import InferenceClient


class CAIAPIInference(LLMInference):
    def __init__(
        cls,
        model_id,
    ):
        from PyCharacterAI import Client



