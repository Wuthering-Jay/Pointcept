"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()


def main():
    
    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"
    config_file = "configs/pointnet2/pointnet2_cls.py"
    options = None
    
    cfg = default_config_parser(config_file, options)
    launch(
        main_worker,
        num_gpus_per_machine=num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        cfg=(cfg,),
    )
    
    # args = default_argument_parser().parse_args()
    # cfg = default_config_parser(args.config_file, args.options)

    # launch(
    #     main_worker,
    #     num_gpus_per_machine=args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     cfg=(cfg,),
    # )


if __name__ == "__main__":
    main()
