import re
from string import ascii_lowercase
from collections import defaultdict

import torch
import numpy as np

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char = self.EMPTY_TOK
        for ind in inds:
            if self.ind2char[ind] == last_char:
                continue
            if self.ind2char[ind] != self.EMPTY_TOK:
                decoded.append(self.ind2char[ind])
            last_char = self.ind2char[ind]
        return ''.join(decoded)

    def ctc_beam_search(self, log_probs, beam_size=20):
        dp = {
            ("", self.EMPTY_TOK): 0.0
        }
        for log_prob in log_probs:
            dp = self._expand_and_merge_path(dp, log_prob)
            dp = self._truncate_paths(dp, beam_size)
        max_prob_prefix = max(dp.items(), key=lambda x: x[1])[0]
        return max_prob_prefix

    def _expand_and_merge_path(self, dp, log_prob):
        new_dp = defaultdict(float)
        for ind, next_token_prob in enumerate(log_prob):
            cur_char = self.ind2char[ind]
            for (prefix, last_char), v in dp.items():
                if last_char == cur_char or cur_char == self.EMPTY_TOK:
                    new_prefix = prefix
                else:
                    new_prefix += cur_char
                new_dp[(new_prefix, cur_char)] = np.log(np.exp(new_dp[(new_prefix, cur_char)]) + np.exp(v + log_prob))
        return new_dp

    def _truncate_paths(self, dp, beam_size):
        return dict(sorted(list(dp.items()), key=lambda x: -x[1])[:beam_size])

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
