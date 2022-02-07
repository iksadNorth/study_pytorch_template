import argparse
import collections
import torch
import numpy as np

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    # logger 객체 생성
    logger = config.get_logger('train')

    # setup data_loader instances
    # 다음은 config.json에 적시된 data_loader를 로드하는 과정이다.
    data_loader = config.init_obj('data_loader', module_data)
    # 아래 코드는 왜 있는지 모르겠음. base_data_loader.py에 있긴한데 정체를 모르겠음.
    # 결과적으로는 valid_data_loader = None 임.
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    # 다음은 config.json에 적시된 model를 로드하는 과정이다.
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    # 다음은 GPU training를 수행하기 위한 사용가능한 gpu 목록을 나열한 것이다.
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    # 다음은 config.json에 적시된 loss 로드하는 과정이다.
    criterion = getattr(module_loss, config['loss'])
    
    # 다음은 config.json에 적시된 metrics 로드하는 과정이다.
    # metrics란, 훈련 중 모델을 평가하기 위한 지표들이다. 여러가지를 나열할 수 있다.
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # requires_grad=True 인 파라미터만 출력
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    
    # 다음은 config.json에 적시된 optimizer 로드하는 과정이다. 
    # optimizer에는 trainable_params가 인자로 들어간다.
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    
    # 다음은 config.json에 적시된 lr_scheduler 로드하는 과정이다.
    # lr_scheduler란 학습을 진행함에 따라 learning rate를 점차 줄이는 방식에 관한 모듈이다.
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    # ex. train.py 
    # --config C:\Users\wjdgn\pytorch_template\config.json
    # --device GPU:1
    # --lr 1e-7
    # --bs 50
    
    
    args = argparse.ArgumentParser(description='PyTorch Template')
    # 다음은 고정적인 arg이다. 여기선 config.json 파일 지정과 사용할 GPU를 지정한다.
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # 다음은 사용자 지정 arg이다.
    # 코드 상 flag는 무조건 --를 붙혀야 한다. -는 허용하지 않는 구조.
    # 코드 상 target은 무조건 ;로 구분해야 함.
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    # 다음 과정을 통해 config 객체에 모든 config 값을 담고 실행시킨다.
    # 다음 매개변수는 
    # args : argparse.ArgumentParser
    # options : List(Nametuple)
    config = ConfigParser.from_args(args, options)
    ###########################################################################################
    # 위의 과정은 config를 사용하기 쉬운 형태로 즉, 객체 형태로 담아두기 위한 과정이다.
    # config : ConfigParser
    main(config)
