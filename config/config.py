from pathlib import Path

BASE_DIR = Path('.')
config = {
    'raw_data_path': BASE_DIR / 'dataset/cnews.txt',
    'test_path': BASE_DIR / 'dataset/test.txt',
    # defined
    'data_dir': BASE_DIR / 'data',
    # defined
    'log_dir': BASE_DIR / 'output/log',
    # defined
    'writer_dir': BASE_DIR / "model/TSboard",
    # defined
    'checkpoint_dir': BASE_DIR / "model/checkpoints",
    'cache_dir': BASE_DIR / 'model/',
    'result': BASE_DIR / "output/result",
    # defined
    'bert_vocab_path': BASE_DIR / 'data/vocab.txt',
    # defined
    'bert_config_file': BASE_DIR / 'model/bert_config.json',
    'bert_model_dir': BASE_DIR / 'model/bert',
}
