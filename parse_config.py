import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # 각 매개변수의 타입은 
        # config : dict
        # resume : Path -> str
        # modification : dict
        # run_id : '%m%d_%H%M%S' -> str / ex. '0207_125607'
        # 정확한건 아닌데 run_id가 꼭 날짜일 필요는 없어보임.
        
        
        # ex. exper_name = "Mnist_LeNet"
        # ex. save_dir = Path('save_dir/')
        
        # self._config : dict
        # self.resume : str
        # self._save_dir = save_dir / 'models' / exper_name / run_id
        # self._log_dir = save_dir / 'log' / exper_name / run_id
        # self.log_levels : dict / {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        
        
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        # 다음은 config.json을 참고하면 됨.
        # ex. save_dir = Path('save_dir/')
        save_dir = Path(self.config['trainer']['save_dir'])

        # ex. exper_name = "Mnist_LeNet"
        exper_name = self.config['name']
        # run_id는 str
        if run_id is None: # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # make directory for saving checkpoints and log.
        # 없는 디렉토리 경로 생성
        # run_id == ''이면 기존에 경로가 있어도 괜찮다는 뜻
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        # 해당 config(option 포함)를 저장함.
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        # logging 환경설정 해주는 곳
        # 기록 수위는 logging.INFO
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        # 다음 매개변수는 
        # args : argparse.ArgumentParser
        # options : List(Nametuple)
        # 다음을 이용해 사용자 지정 arg을 CLI 위로 올려놓는다.
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        # 다음은 CLI 파라미터를 args 객체 위에 올린다.
        if not isinstance(args, tuple):
            args = args.parse_args()

        # 다음은 지정한 GPU를 실제로 지정하는 과정.
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        # resume이 있으면 resume에 담긴 config.json을 불러온다.
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        # resume이 없으면 config에 지정한 위치에 담긴 config.json을 불러온다.
        # resume이 없고 config도 없으면 오류 발생 "Configuration file need to be specified. Add '-c config.json', for example."
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        # 결국 이 과정을 통해 
        # 'cfg_fname = {config.json 파일이 있는 경로, Path 객체}'
        # 'resume = None or {resume.json 파일이 있는 경로, Path 객체}'
        
        # 다음은 config.json을 json 객체로 변환
        config = read_json(cfg_fname)
        # 다음은 resume이 있으면 resume을 위의 json 객체에 업데이트 시킴.
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        # ex. modification = {'optimizer;args;lr' : args.lr}
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        # 각 매개변수의 타입은 
        # config : dict
        # resume : Path
        # modification : dict
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        
        # 모듈 자체를 매개변수로 전달할 수 있는지는 또 첨앎.
        # data_loaders.py에 있는 클래스 중 config.json에 있는 것을 참고하여 사용함.
        # 주어진 매개변수 외에도 config.json에 있는 매개변수을 얹어서 사용하게 됨.
        # dataLoader 객체를 출력
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        
        # 주어진 매개변수 외에도 config.json에 있는 매개변수을 얹어서 사용하게 됨.
        # 본 메소드에 전달된 매개변수 + config.json에 있는 매개변수를 아예 함수에 박제해둠.
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions to update config dict with custom cli options
# modification을 좀더 정제해서 config에 합치는 메소드
# modification이 None이면 그대로 config만 return
# modification 값 중 None이면 누락시킴
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

# 걍 앞에 --를 떼어 준다고 생각하면 됨.
# '--version' -> 'version'
def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

# 다음은 ;으로 경계를 나눈 다음 config 딕셔너리에서 해당 키로 접근하는 과정이다. 
# 'optimizer;args;lr'를 예시로 든다.
def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(';')
    # keys = ['optimizer', 'args','lr']
    _get_by_path(tree, keys[:-1])[keys[-1]] = value
    # _get_by_path(config, ['optimizer', 'args'])['lr'] = 1e-7

# 다음은 config 딕셔너리에서 해당 키로 접근하는 과정이다. 
def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
