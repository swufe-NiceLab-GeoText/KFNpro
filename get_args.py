import json
import utils


class Args:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.dataset = config["dataset"]
        self.train_path = config["train_path"]
        self.dev_path = config["dev_path"]
        self.test_path = config["test_path"]
        self.checkpoint_path = config["checkpoint_path"]
        self.types_dict = None

        self.epochs = config["epochs"]
        self.episodes = config["episodes"]
        self.episodes_q = config["episodes_q"]
        self.numFreeze = config["numFreeze"]
        self.filevocab = config["filevocab"]
        self.fileModelConfig = config["fileModelConfig"]
        self.fileModel = config["fileModel"]

        self.gpu = config["gpu"]
        self.seed = config["seed"]
        self.hidden_size = config["hidden_size"]
        self.nway = config["nway"]
        self.kshot = config["kshot"]
        self.qshot = config["qshot"]
        self.text_max_len = config["text_max_len"]
        self.label_max_len = config["label_max_len"]
        self.beta = config["beta"]
        self.step = config["step"]
        self.random_len = config["random_len"]
        self.it_num = config["it_num"]
        self.temprature = config["temprature"]
        self.temprature1 = config["temprature1"]
        self.alpha = config["alpha"]
        self.gama = config["gama"]
        self.se_layer = config["se_layer"]
        self.q_qshot = config["q_qshot"]

        self.prompt_len = config["prompt_len"]
        self.key_hidden_size = config["key_hidden_size"]
        self.pool_len = config["pool_len"]
        self.key_init_method = config["key_init_method"]

        self.optionn = "bert_layer"
        self.dataset_name = "none"

        self.weight_decay = config["weight_decay"]
        self.learning_rate = config["learning_rate"]
        self.warmup_steps = config["warmup_steps"]
        self.dropout = config["dropout"]

        # LLM / LoRA settings (optional, with defaults for backward compatibility)
        self.use_lora = config.get("use_lora", False)
        self.lora_r = config.get("lora_r", 16)
        self.lora_alpha = config.get("lora_alpha", 32)
        self.lora_dropout = config.get("lora_dropout", 0.1)
        self.fp16 = config.get("fp16", False)

        self.gama1 = 1.0
        self.gama2 = 1.0

        self.logger_path = "code/"
        self.logger = utils.get_logger("aaa", self.logger_path)
        self.logger.info("Initialized config from: " + path)

    def __repr__(self):
        return "{}".format(self.__dict__.items())

    def show_self(self):
        print("=" * 50)
        print("Configuration:")
        print("=" * 50)
        for key in ['train_path', 'dev_path', 'test_path', 'checkpoint_path',
                    'epochs', 'episodes', 'numFreeze', 'filevocab', 'fileModel',
                    'gpu', 'seed', 'hidden_size', 'nway', 'kshot', 'qshot',
                    'text_max_len', 'label_max_len', 'key_hidden_size',
                    'pool_len', 'key_init_method', 'prompt_len',
                    'weight_decay', 'learning_rate', 'warmup_steps']:
            print(f"  {key}: {getattr(self, key)}")
        print("=" * 50)

    def write_self(self, path):
        """Save configuration as a proper JSON file."""
        config = {
            'train_path': self.train_path,
            'dev_path': self.dev_path,
            'test_path': self.test_path,
            'checkpoint_path': self.checkpoint_path,
            'epochs': self.epochs,
            'episodes': self.episodes,
            'episodes_q': self.episodes_q,
            'numFreeze': self.numFreeze,
            'filevocab': self.filevocab,
            'fileModelConfig': self.fileModelConfig,
            'fileModel': self.fileModel,
            'gpu': self.gpu,
            'seed': self.seed,
            'hidden_size': self.hidden_size,
            'nway': self.nway,
            'kshot': self.kshot,
            'qshot': self.qshot,
            'text_max_len': self.text_max_len,
            'label_max_len': self.label_max_len,
            'beta': self.beta,
            'step': self.step,
            'random_len': self.random_len,
            'it_num': self.it_num,
            'temprature': self.temprature,
            'temprature1': self.temprature1,
            'alpha': self.alpha,
            'gama': self.gama,
            'se_layer': self.se_layer,
            'q_qshot': self.q_qshot,
            'prompt_len': self.prompt_len,
            'key_hidden_size': self.key_hidden_size,
            'pool_len': self.pool_len,
            'key_init_method': self.key_init_method,
            'weight_decay': self.weight_decay,
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'dropout': self.dropout,
            'optionn': self.optionn,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def update_from_args(self, cli_args):
        """Update config from command-line arguments. Sentinel values are skipped."""
        mapping = {
            'gpu': ('gpu', -1),
            'tempra1': ('temprature1', -1.0),
            'kshot': ('kshot', -1),
            'beta': ('beta', -1.0),
            'temprature': ('temprature', -1.0),
            'dataset_num': ('dataset', "09"),
            'seed': ('seed', -1),
            'qshot': ('qshot', -1),
            'alpha': ('alpha', -1.0),
            'gama': ('gama', -1.0),
            'se_layer': ('se_layer', -1),
            'weight_decay': ('weight_decay', 0.00),
            'margin': ('margin', -1.0),
            'prompt_len': ('prompt_len', -1),
            'numFreeze': ('numFreeze', -1),
            'pool_len': ('pool_len', -1),
            'epochs': ('epochs', -1),
            'warmup_steps': ('warmup_steps', -1),
            'step': ('step', -1),
            'learning_rate': ('learning_rate', -1.0),
            'dropout': ('dropout', -1.0),
            'text_len': ('text_max_len', -1),
            'label_len': ('label_max_len', -1),
            'gama1': ('gama1', -1.0),
            'gama2': ('gama2', -1.0),
        }
        for cli_name, (attr_name, sentinel) in mapping.items():
            cli_val = getattr(cli_args, cli_name, sentinel)
            if cli_val != sentinel:
                setattr(self, attr_name, cli_val)

        # Always-set fields
        self.optionn = cli_args.optionn
        self.dataset_name = cli_args.dataset_name
