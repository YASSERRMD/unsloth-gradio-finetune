from typing import List
from datasets import Dataset as HFDataset
from .schema import CanonicalRecord


def render_for_training(
    records: List[CanonicalRecord], tokenizer, add_eos: bool = True
) -> HFDataset:
    texts = []
    for rec in records:
        if rec.meta.get("text"):
            texts.append(rec.meta["text"])
        else:
            msg_dicts = [{"role": m.role, "content": m.content} for m in rec.messages]
            text = tokenizer.apply_chat_template(
                msg_dicts, tokenize=False, add_generation_prompt=False
            )
            if add_eos and not text.endswith(tokenizer.eos_token):
                text += tokenizer.eos_token
            texts.append(text)
    return HFDataset.from_dict({"text": texts})
