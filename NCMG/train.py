import os
# os.environ["CUDA_VISIBLE_DEVICES"]='8'
import torch
from model import NCMG
from dataload import args, train_loader,valid_loader,test_loader
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch + "_mhgnn_u_6.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)
        
        
model = NCMG(args,args.get("num_users"),args.get("feature_user"),args.get("feature_user_visual"),1,
                   args.get("feature_BS"),args.get("feature_BS_visual"),1,args.get("feature_IRS"),args.get("feature_IRS_visual"),
               args.get("hidden"),args.get("dropout"))

model.to(device)
    
    

optimizer = torch.optim.Adam(params=model.parameters(), lr=args.get('lr_init'), eps=1.0e-8,
                             weight_decay=1e-5, amsgrad=False)
#learning rate decay
lr_scheduler = None
if args.get('lr_decay'):
    print('Applying learning rate decay.')
    #r_decay_steps = [int(i) for i in args.get('lr_decay_step')]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
    milestones=[0.33 * args.get('epochs'),0.5 * args.get('epochs'),0.8 * args.get('epochs'),0.9 * args.get('epochs')],gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=64)



#start training
trainer = Trainer(model, optimizer, train_loader, valid_loader, test_loader, args, lr_scheduler)

trainer.train()

# result_train_file = os.path.join("RIS-MIMO-THz")
    
    
# save_model(trainer,result_train_file,1)
# 保存模型
save_model(model, args.get("model_dir"), 1)
print(f"模型保存到: {args.get('model_dir')}")

print("Starting testing...")
total_sum,rate,power,phase,band=trainer.test()

# print(f"Average Sum Rate: {sum(total_sum)/len(total_sum):.4f}")
# print(f"Final Rate Shape: {rate.shape}")
# print(f"Final Beamforming Shape: {power.shape}")
# print(f"Final Phase Shape: {phase.shape}")
# print(f"Final Bandwidth Shape: {band.shape}")

