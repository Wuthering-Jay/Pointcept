import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():

    num_gpus = 1
    num_machines = 1
    machine_rank = 0
    dist_url = "auto"
    config_file = "configs/opengf/semseg-pt-v2m2-0-base.py"
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
