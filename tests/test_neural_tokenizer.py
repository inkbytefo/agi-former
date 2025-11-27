## Developer: inkbytefo
## Modified: 2025-11-27

from src.tokenizer.neural_tokenizer import build_char_vocab, encode_word_chars, train_distill
from src.data.morphology import build_vocab

def test_train_distill_runs_and_analyzes():
    texts = ["Türkiye Cumhuriyeti 1923 yılında kuruldu.", "İstanbul Türkiye'nin en kalabalık şehridir."]
    root2id, suffix2id = build_vocab(texts, root_limit=1000, suffix_limit=100)
    analyzer = train_distill(texts, lambda w: (w.lower(), []), root2id, suffix2id, suffix_slots=3, epochs=1)
    r, sfx = analyzer("Türkiye")
    assert isinstance(r, str)
    assert isinstance(sfx, list)

