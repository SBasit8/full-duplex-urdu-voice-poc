# train_tokenizer.py

import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='ur.txt',
    model_prefix='urdu_tokenizer',
    vocab_size=32000,
    model_type='bpe',
    character_coverage=0.9995,
    input_sentence_size=5000000,
    shuffle_input_sentence=True,
    pad_id=3,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
    user_defined_symbols=[],
    num_threads=8
)

print("Tokenizer training complete.")
print("Files created: urdu_tokenizer.model, urdu_tokenizer.vocab")