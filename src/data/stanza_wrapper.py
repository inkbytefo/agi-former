## Developer: inkbytefo
## Modified: 2025-11-27

import stanza

class StanzaAnalyzer:
    def __init__(self, lang: str = "tr", use_gpu: bool = True):
        self.lang = lang
        self.use_gpu = use_gpu
        self.nlp = None

    def start(self) -> bool:
        try:
            stanza.download(self.lang, verbose=False)
            self.nlp = stanza.Pipeline(self.lang, processors="tokenize,pos,lemma", use_gpu=self.use_gpu, tokenize_pretokenized=False)
            return True
        except Exception:
            self.nlp = None
            return False

    def _ud_to_suffixes(self, feats: str):
        if not feats:
            return []
        parts = str(feats).split("|")
        sfx = []
        for p in parts:
            if p.startswith("Number=Plur"):
                sfx.append("+LAR")
            elif p.startswith("Case=Dat"):
                sfx.append("+DA")
            elif p.startswith("Case=Loc"):
                sfx.append("+DE")
            elif p.startswith("Case=Abl"):
                sfx.append("+DAN")
            elif p.startswith("Case=Acc"):
                sfx.append("+I")
        return sfx

    def analyze(self, word: str):
        if self.nlp is None:
            return word.lower(), []
        doc = self.nlp(word)
        for sent in doc.sentences:
            for w in sent.words:
                root = w.lemma.lower() if w.lemma else word.lower()
                sfx = self._ud_to_suffixes(w.feats)
                return root, sfx
        return word.lower(), []

