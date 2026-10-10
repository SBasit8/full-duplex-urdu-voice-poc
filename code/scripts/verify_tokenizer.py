# verify_tokenizer.py

import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load('urdu_tokenizer.model')

test_sentences = [
    "آپ کا نام کیا ہے؟",
    "میں آپ کی مدد کر سکتا ہوں",
    "یہ بہت اچھا سوال ہے",
    "Pakistan mein urdu boli jaati hai", # code-switching test
    "yaar sun, basically main ye keh raha hoon",
    "آپ ka plan kya hai, basically target kya kar rahe hain",
    "mere liye ek meeting schedule karo", 
]

for s in test_sentences:
    tokens = sp.encode(s, out_type=str)
    ids = sp.encode(s, out_type=int)
    print(f"\nInput:  {s}")
    print(f"Tokens: {tokens}")
    print(f"Count:  {len(ids)} tokens")
    decoded = sp.decode(ids)
    print(f"Decoded: {decoded}")