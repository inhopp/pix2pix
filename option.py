import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--multigpu", type=bool, default=True)
    parser.add_argument("--device", type=str, default="0")

    # models
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--input_size", type=int, default=256)
    
    # dataset
    parser.add_argument("--data_dir", type=str, default="./datasets/")

    # training setting
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--l1_lambda", type=float, default=100)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    
    # misc
    parser.add_argument("--ckpt_root", type=str, default="./FT_model")

    return parser.parse_args()


def make_template(opt):
    # device
    opt.device_ids = [int(item) for item in opt.device.split(',')]
    if len(opt.device_ids) == 1:
        opt.multigpu = False
    opt.gpu = opt.device_ids[0]


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt