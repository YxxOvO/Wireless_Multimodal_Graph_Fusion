import os
# os.environ["CUDA_VISIBLE_DEVICES"]='8'
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model import NCMG
from dataload import args, train_loader, valid_loader, test_loader
from trainer import Trainer

# Get rank info for distributed training
def get_rank():
    return int(os.environ.get('RANK', 0))

def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))

def is_main_process():
    return get_rank() == 0

def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check for DDP and quantization flags from args
use_ddp = args.get('ddp', False)
use_quantize = args.get('quantize', False)
local_rank = 0

# Setup DDP if enabled
if use_ddp:
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if device.type == 'cuda':
        torch.cuda.set_device(local_rank)
    print_main(f"[Rank {get_rank()}] DDP enabled, local_rank={local_rank}, world_size={get_world_size()}")


model = NCMG(args,args.get("num_users"),args.get("feature_user"),args.get("feature_user_visual"),1,
                   args.get("feature_BS"),args.get("feature_BS_visual"),1,args.get("feature_IRS"),args.get("feature_IRS_visual"),
               args.get("hidden"),args.get("dropout"))

model.to(device)

# Wrap model with DDP if enabled
if use_ddp and device.type == 'cuda':
    model = DDP(model, device_ids=[local_rank])

# Apply dynamic int8 quantization for inference if enabled (and NOT training)
if use_quantize and not use_ddp:
    print_main("Applying int8 dynamic quantization for inference...")
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.eval()

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr_init'), eps=1.0e-8,
                             weight_decay=1e-5, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.get('lr_decay'):
    print_main('Applying learning rate decay.')
    #r_decay_steps = [int(i) for i in args.get('lr_decay_step')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    milestones=[0.33 * args.get('epochs'),0.5 * args.get('epochs'),0.8 * args.get('epochs'),0.9 * args.get('epochs')],gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch + "_mhgnn_u_6.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)


# Synchronize before training starts (for DDP)
if use_ddp and device.type == 'cuda':
    dist.barrier()

#start training
trainer = Trainer(model, optimizer, train_loader, valid_loader, test_loader, args, lr_scheduler)

trainer.train()

# Synchronize after training before saving
if use_ddp and device.type == 'cuda':
    dist.barrier()

# result_train_file = os.path.join("RIS-MIMO-THz")


# save_model(trainer,result_train_file,1)
# 保存模型
if is_main_process():
    save_model(model, args.get("model_dir"), 1)
    print_main(f"模型保存到: {args.get('model_dir')}")

# Synchronize before testing
if use_ddp and device.type == 'cuda':
    dist.barrier()

print_main("Starting testing...")
total_sum,rate,power,phase,band=trainer.test()

# print(f"Average Sum Rate: {sum(total_sum)/len(total_sum):.4f}")
# print(f"Final Rate Shape: {rate.shape}")
# print(f"Final Beamforming Shape: {power.shape}")
# print(f"Final Phase Shape: {phase.shape}")
# print(f"Final Bandwidth Shape: {band.shape}")