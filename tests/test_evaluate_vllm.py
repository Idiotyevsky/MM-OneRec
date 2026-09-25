import importlib.util
import sys
from types import SimpleNamespace


def _module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_vllm_test", "scripts/evaluate_vllm.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Logprob:
    def __init__(self, value):
        self.logprob = value


class _FakeEngine:
    def __init__(self):
        self.allowed_history = []
        self.scores = {
            10: -0.10,
            11: -0.30,
            20: -0.10,
            21: -0.50,
            22: -0.10,
            99: -0.01,
        }

    def generate(self, prompts, sampling_params, use_tqdm):
        assert use_tqdm is False
        outputs = []
        for prompt, params in zip(prompts, sampling_params):
            allowed = tuple(params["allowed"])
            self.allowed_history.append(allowed)
            logprobs = {
                token_id: _Logprob(self.scores[token_id])
                for token_id in allowed
            }
            outputs.append(
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            token_ids=[max(logprobs, key=lambda key: self.scores[key])],
                            logprobs=[logprobs],
                        )
                    ]
                )
            )
        return outputs


class _Tokenizer:
    eos_token_id = 99

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        mapping = {10: "<a_0>", 11: "<a_1>", 20: "<b_0>\n", 21: "<b_1>\n", 22: "<b_2>\n"}
        return "".join(mapping[token_id] for token_id in token_ids)


def test_catalog_trie_masks_each_beam_and_returns_ranked_valid_paths():
    module = _module()
    trie = module.CatalogTrie(
        [
            [10, 20, 99],
            [10, 21, 99],
            [11, 22, 99],
        ]
    )
    engine = _FakeEngine()

    beams = module.constrained_beam_search(
        engine=engine,
        prompt_token_ids=[[1, 2]],
        trie=trie,
        eos_token_id=99,
        beam_width=2,
        max_new_tokens=4,
        length_penalty=0.0,
        sampling_params_factory=lambda allowed, width: {
            "allowed": list(allowed),
            "width": width,
        },
    )

    assert [beam.token_ids for beam in beams[0]] == [
        (10, 20, 99),
        (11, 22, 99),
    ]
    assert all(trie.contains(beam.token_ids) for beam in beams[0])
    assert (10, 11) in engine.allowed_history
    assert (20, 21) in engine.allowed_history
    assert (22,) in engine.allowed_history


def test_decode_beams_strips_eos_and_response_newline():
    module = _module()
    rows = [[module.Beam((10, 20, 99), -0.2)]]
    assert module.decode_beams(_Tokenizer(), rows) == [["<a_0><b_0>"]]

