"""One-time fix: make AttnGrounder's spaCy-2.x corpus.pth loadable under spaCy 3.x.

corpus.pth (~487MB) embeds a spaCy 2.x `Language` object that references
`spacy.lemmatizer` (removed in spaCy 3.x), so `torch.load` fails with
`ModuleNotFoundError: No module named 'spacy.lemmatizer'`.

We only need two things from the corpus, and neither needs the 2.x pipeline:
  * `dictionary` (token -> index)  -- MUST stay identical to the trained embedding
  * `glove` vectors               -- used to init the embedding (overwritten at eval)
The tokenizer just splits surface tokens; `spacy.blank("en")` uses the SAME English
tokenizer rules as en_core_web_sm, so tokens reproduce.

This loads corpus.pth with a spaCy-agnostic unpickler (stubbing every spaCy/thinc
class to an inert dummy), swaps `nlp` for `spacy.blank("en")`, and re-saves so
AttnGrounder's loader (`torch.load(corpus.pth)`) works unchanged.

Run once on the remote, from the AttnGrounder repo root:
    python analysis/fix_corpus.py --corpus ln_data/corpus.pth
"""
import os
import sys
import types
import shutil
import pickle
import argparse

import torch

# packages whose classes we stub out during unpickling (spaCy 2.x internals + deps)
_STUB_PKGS = {
    "spacy", "thinc", "blis", "cymem", "preshed", "murmurhash", "srsly",
    "catalogue", "wasabi", "plac", "ml_datasets", "spacy_legacy", "spacy_loggers",
    "pydantic", "cython",
}


class _DummyMeta(type):
    """Some pickled state calls the REAL builtin `getattr(SomeStubbedClass, 'attr')`
    directly (builtins isn't in _STUB_PKGS, so that call is never intercepted by us).
    A plain class has no such attribute -> AttributeError. This metaclass makes
    CLASS-level attribute lookups on _Dummy degrade to another dummy instead."""
    def __getattr__(cls, name):
        return _Dummy


class _Dummy(metaclass=_DummyMeta):
    """Absorbs any pickle reconstruction protocol a stubbed spaCy/thinc object
    might use: plain __init__/__new__, __setstate__, SETITEMS (dict/list-like via
    __reduce__'s listitems/dictitems -> __setitem__/append), arbitrary attribute
    get/set (instance AND class level, via __getattr__ + the metaclass above), and
    being called as if it were a function/method reference."""
    def __init__(self, *a, **k): pass
    def __new__(cls, *a, **k): return object.__new__(cls)
    def __setstate__(self, state): pass
    def __setitem__(self, k, v): pass
    def __getitem__(self, k): return _Dummy()
    def __setattr__(self, k, v): pass
    def __getattr__(self, k): return _Dummy()   # unknown instance attr -> dummy (also callable)
    def append(self, v): pass
    def extend(self, v): pass
    def __call__(self, *a, **k): return []
    def __len__(self): return 0
    def __iter__(self): return iter([])
    def __reduce__(self):
        return (_Dummy, ())


def _dummy_factory():
    return _Dummy


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in _STUB_PKGS:
            return _Dummy
        return super().find_class(module, name)


def _stub_pickle_module():
    m = types.ModuleType("corpus_stub_pickle")
    m.Unpickler = _StubUnpickler
    m.load, m.loads = pickle.load, pickle.loads
    m.Pickler, m.dump, m.dumps = pickle.Pickler, pickle.dump, pickle.dumps
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="ln_data/corpus.pth")
    args = ap.parse_args()

    # so the unpickler can resolve utils.word_utils.{Corpus,Dictionary}.
    # Insert cwd (repo root) ONLY -- do NOT add cwd/utils, that shadows the
    # `utils` package with the `utils/utils.py` module ("utils is not a package").
    cwd = os.getcwd()
    sys.path.insert(0, cwd)

    print("[1] loading old corpus with a spaCy-agnostic unpickler ...")
    corpus = torch.load(args.corpus, pickle_module=_stub_pickle_module(), weights_only=False)
    vocab = len(corpus.dictionary)
    has_glove = getattr(corpus, "glove", None) not in (None, {})
    print(f"    recovered {type(corpus).__name__}: vocab={vocab}  glove_present={has_glove}")

    if not hasattr(corpus, "glove") or corpus.glove is None:
        # keep get_glove_embed() from crashing; missing words just random-init
        corpus.glove = {}
        print("    WARNING: no GloVe vectors in corpus -> embedding init will be random "
              "(fine for eval; for from-scratch training supply glove_dict_6B.300.pkl)")

    print("[2] installing modern tokenizer spacy.blank('en') ...")
    import spacy
    corpus.nlp = spacy.blank("en")

    ids = corpus.tokenize("turn left behind the silver car")
    toks = [corpus.dictionary[int(i)] for i in ids[:8]]
    print(f"    tokenize OK -> ids{tuple(ids.shape)} first tokens: {toks}")

    bak = args.corpus + ".spacy2.bak"
    if not os.path.exists(bak):
        shutil.copyfile(args.corpus, bak)
        print(f"    backed up original -> {bak}")
    torch.save(corpus, args.corpus)
    print(f"[3] re-saved modern corpus -> {args.corpus}  (loader will now work unchanged)")


if __name__ == "__main__":
    main()
