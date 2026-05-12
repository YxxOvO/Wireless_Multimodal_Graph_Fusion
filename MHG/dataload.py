import torch
import torch.utils.data
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dBm_watt(x):
    
    return 10**(x/10)/1000

# args={"IRS_elements":9,"IRS_elements_row":3,"BS_row":2,"BS_col":3,"BS_antenna":6,"num_users":6,"P_max":dBm_watt(30),"noise_pow":dBm_watt(-174),
#       "loc_BS":torch.tensor((25,-20,-5)),"loc_IRS":torch.tensor((0,0,0)),"user_range_x1":(0,15),"user_range_x2":(-2.5,-10),
#       "user_range_y":(0,25),"node_types":3,"sub_bands":5,"f_start":0.380e12,"f_end":0.4e12,"r_u_min":13e9,"user_antenna":2,
#       "hidden":1024,"dropout":0.005,"eta_1":111.48,"eta_2":-2.97e-10,"eta_3":0.01,"int_samp":100000,"lam1":11e9,"lam2":11e9,
#       "user_range_z":-10,"Rician_factor":1,"b_g":0.75e9,"b_max":4e9,"antenna_space":395e-6,
#       "IRS_space":395e-6,"inp_dim":2,"hid_dim":4,"num_metapath":5,"train_s":400,"validation_s":20,
#       "test_s":177,"samples":597,"batch":1,"lr_init":0.0005,"epochs":250,"log_step":100,"lr_decay":True}
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="配置超参数")
    
    # 添加超参数
    parser.add_argument('--IRS_elements', default=64, type=int, help='IRS elements')
    parser.add_argument('--IRS_elements_row', default=8, type=int, help='IRS elements row')
    parser.add_argument('--BS_row', default=2, type=int, help='BS row')
    parser.add_argument('--BS_col', default=10, type=int, help='BS col')
    parser.add_argument('--BS_antenna', default=20, type=int, help='BS antennas')
    parser.add_argument('--num_users', default=6, type=int, help='Number of users')
    parser.add_argument('--P_max', default=dBm_watt(30), type=float, help='Max power in watts')
    parser.add_argument('--noise_pow', default=dBm_watt(-174), type=float, help='Noise power in watts')
    parser.add_argument('--loc_BS', default=torch.tensor((25, -20, -5)), type=torch.Tensor, help='Location of BS')
    parser.add_argument('--loc_IRS', default=torch.tensor((0, 0, 0)), type=torch.Tensor, help='Location of IRS')
    parser.add_argument('--user_range_x1', default=(0, 15), type=tuple, help='User range x1')
    parser.add_argument('--user_range_x2', default=(-2.5, -10), type=tuple, help='User range x2')
    parser.add_argument('--user_range_y', default=(0, 25), type=tuple, help='User range y')
    parser.add_argument('--node_types', default=3, type=int, help='Node types')
    parser.add_argument('--sub_bands', default=5, type=int, help='Sub bands')
    parser.add_argument('--f_start', default=0.380e12, type=float, help='Start frequency')
    parser.add_argument('--f_end', default=0.4e12, type=float, help='End frequency')
    parser.add_argument('--r_u_min', default=13e9, type=float, help='Min user rate')
    parser.add_argument('--user_antenna', default=2, type=int, help='User antenna')
    parser.add_argument('--hidden', default=1024, type=int, help='Hidden layer size')
    parser.add_argument('--dropout', default=0.005, type=float, help='Dropout rate')
    parser.add_argument('--eta_1', default=111.48, type=float, help='Eta 1')
    parser.add_argument('--eta_2', default=-2.97e-10, type=float, help='Eta 2')
    parser.add_argument('--eta_3', default=0.01, type=float, help='Eta 3')
    parser.add_argument('--int_samp', default=10000, type=int, help='Intermediate sample size')
    parser.add_argument('--lam1', default=11e9, type=float, help='Lambda 1')
    parser.add_argument('--lam2', default=11e9, type=float, help='Lambda 2')
    parser.add_argument('--user_range_z', default=-10, type=float, help='User range z')
    parser.add_argument('--Rician_factor', default=1, type=float, help='Rician factor')
    parser.add_argument('--b_g', default=0.75e9, type=float, help='Base G')
    parser.add_argument('--b_max', default=4e9, type=float, help='Max B')
    parser.add_argument('--antenna_space', default=395e-6, type=float, help='Antenna space')
    parser.add_argument('--IRS_space', default=395e-6, type=float, help='IRS space')
    parser.add_argument('--inp_dim', default=2, type=int, help='Input dimension')
    parser.add_argument('--hid_dim', default=4, type=int, help='Hidden dimension')
    parser.add_argument('--num_metapath', default=5, type=int, help='Number of metapath')
    parser.add_argument('--train_s', default=400, type=int, help='Training samples')
    parser.add_argument('--validation_s', default=20, type=int, help='Validation samples')
    parser.add_argument('--test_s', default=177, type=int, help='Test samples')
    parser.add_argument('--samples', default=597, type=int, help='Samples')
    parser.add_argument('--batch', default=1, type=int, help='Batch size')
    parser.add_argument('--lr_init', default=0.0005, type=float, help='Initial learning rate')
    parser.add_argument('--epochs', default=250, type=int, help='Number of epochs')
    parser.add_argument('--log_step', default=100, type=int, help='Log step')
    parser.add_argument('--lr_decay', action='store_true', default=True, help='Learning rate decay')

    # 添加日志目录和模型保存路径的命令行参数
    parser.add_argument('--log_dir', default='logs_train', type=str, help='日志保存路径')
    parser.add_argument('--model_dir', default='./saved_models', type=str, help='模型保存路径')

    return parser.parse_args()

# 使用 argparse 解析命令行参数
args = parse_args()

# 将命令行参数组织成一个字典
args = vars(args)
# 打印参数
print("使用的参数:", args)



#half of users one side, the other half on the other side (always even)

#x~[5,30],y~[-15,25],z=-10

#x~[-5,-15],y~[-15,25],z=-10


def dist(a,b):
    
    return torch.sqrt(torch.sum((a-b)*(a-b)))



d_BR=dist(args.get("loc_BS"),args.get("loc_IRS"))

cos_phi_cos_theta_BR=abs(args.get("loc_IRS")[0]-args.get("loc_BS")[0])/d_BR

sin_phi_cos_theta_BR=abs(args.get("loc_IRS")[1]-args.get("loc_BS")[1])/d_BR

sin_theta_BR=abs(args.get("loc_IRS")[2]-args.get("loc_BS")[2])/d_BR

BS_antennas=range(args.get("BS_antenna"))

user_antennas=range(args.get("user_antenna"))

angles_R=torch.zeros(args.get("IRS_elements"))

angles_B=torch.zeros(args.get("BS_antenna"))

#angles_B=cos_phi_cos_theta_BR*torch.tensor(BS_antennas)

for l in range(args.get("IRS_elements")):
            
    angles_R[l]=(l%args.get("IRS_elements_row"))*sin_phi_cos_theta_BR+math.floor(l/args.get("IRS_elements_row"))*sin_theta_BR

    
for n in range(args.get("BS_antenna")):
            
    angles_B[n]=(n%args.get("BS_row"))*sin_phi_cos_theta_BR+math.floor(n/args.get("BS_col"))*sin_theta_BR



args["angles_R"]=angles_R.to(device)

args["angles_B"]=angles_B.to(device)










user_pos=torch.zeros(args.get("samples"),args.get("num_users"),3)


X_user=torch.zeros(args.get("samples"),args.get("num_users"),3)
### angles user-to-irs    U*L^2


angles_uR=torch.zeros(args.get("samples"),args.get("num_users"),args.get("IRS_elements"))

### angles user-to-bs    U*N_t

angles_uB=torch.zeros(args.get("samples"),args.get("num_users"),args.get("BS_antenna"))

angles_uBR=torch.zeros(args.get("samples"),args.get("num_users"),2,args.get("user_antenna"))

dist_uB=torch.zeros(args.get("samples"),args.get("num_users"),1)

dist_uR=torch.zeros(args.get("samples"),args.get("num_users"),1)

for k in range(args.get("samples")):
    
    
    
    user_pos[k,:,2]=args.get("user_range_z")
    
    user_pos[k,:,0]=(args.get("user_range_x1")[1]-args.get("user_range_x1")[0])*torch.rand(1,int(args.get("num_users")))+args.get("user_range_x1")[0]
    
    #user_pos[k,int(args.get("num_users")/2):,0]=(args.get("user_range_x2")[1]-args.get("user_range_x2")[0])*torch.rand(1,int(args.get("num_users")/2))+args.get("user_range_x2")[0]
    
    user_pos[k,:,1]=(args.get("user_range_y")[1]-args.get("user_range_y")[0])*torch.rand(1,int(args.get("num_users")))+args.get("user_range_y")[0]

    
    for u in range(args.get("num_users")):
        
        
        d_uR=dist(user_pos[k,u,:],args.get("loc_IRS"))
        
        d_uB=dist(user_pos[k,u,:],args.get("loc_BS"))
        
        dist_uB[k,u,0]=d_uB
        
        dist_uR[k,u,0]=d_uR
        
        X_user[k,u,0]=d_uB
        
        X_user[k,u,1]=d_uR

        X_user[k,u,2]=args.get("r_u_min")/1e9
        
        sin_phi_cos_theta_uR=abs(user_pos[k,u,1]-args.get("loc_IRS")[1])/d_uR
        
        sin_theta_uR=abs(user_pos[k,u,2]-args.get("loc_IRS")[2])/d_uR
        
        
        sin_phi_cos_theta_uB=abs(user_pos[k,u,1]-args.get("loc_BS")[1])/d_uB
        
        sin_theta_uB=abs(user_pos[k,u,2]-args.get("loc_BS")[2])/d_uB
        
        
        cos_phi_cos_theta_uB=abs(user_pos[k,u,0]-args.get("loc_BS")[0])/d_uB
        
        
        cos_phi_cos_theta_uBR0=abs(user_pos[k,u,0]-args.get("loc_IRS")[0])/d_uR
        
        
        angles_uBR[k,u,0,:]=cos_phi_cos_theta_uB*torch.tensor(user_antennas)
        
        angles_uBR[k,u,1,:]=cos_phi_cos_theta_uBR0*torch.tensor(user_antennas)
        
        for l in range(args.get("IRS_elements")):
            
            
            angles_uR[k,u,l]=(l%args.get("IRS_elements_row"))*sin_phi_cos_theta_uR+math.floor(l/args.get("IRS_elements_row"))*sin_theta_uR
        
        
        for n in range(args.get("BS_antenna")):
            
            angles_uB[k,u,n]=(n%args.get("BS_row"))*sin_phi_cos_theta_uB+math.floor(n/args.get("BS_col"))*sin_theta_uB
        
        
        
        #angles_uB[k,u,:]=cos_phi_cos_theta_uB*torch.tensor(BS_antennas)
        
        
args["angles_uR"]=angles_uR.to(device)
args["angles_uB"]=angles_uB.to(device)

args["dist_ub"]=dist_uB
args["dist_ur"]=dist_uR
args["dist_br"]=d_BR

args["stream"]=math.floor(min(args.get("BS_antenna"),args.get("num_users")*args.get("user_antenna"))/args.get("num_users"))



adj_user_BS=F.softmax(dist_uB.pow_(-1),dim=1)

adj_user_IRS=F.softmax(dist_uR.pow_(-1),dim=1)

adj_BS_IRS=torch.zeros(1,1)

adj_BS_IRS[0,0]=1/d_BR

adj_BS_IRS=F.softmax(adj_BS_IRS,dim=1)

dist_BR=d_BR*torch.ones(args.get("samples"),1)


X_BS=torch.cat((dist_uB[:,:,0],dist_BR),1)

X_BS=X_BS/torch.max(X_BS)

X_IRS=torch.cat((dist_uR[:,:,0],dist_BR),1)

X_IRS=X_IRS/torch.max(X_IRS)

X_BS=X_BS[:,None,:]

X_IRS=X_IRS[:,None,:]




#meta_paths



#1 U
X_user_U=X_user/torch.max(X_user)


#2 UR

X_user_UR=torch.einsum('bij,bjk->bik',adj_user_IRS, X_IRS)

#3 UB

X_user_UB=torch.einsum('bij,bjk->bik',adj_user_BS, X_BS)

#4 RB (irs)

X_IRS_RB=torch.einsum('ij,bjk->bik',adj_BS_IRS, X_BS)

#5 URB

X_user_URB=torch.einsum('bij,bjk->bik',adj_user_IRS, X_IRS_RB)

#6 BR (BS)

X_BS_BR=torch.einsum('ij,bjk->bik',adj_BS_IRS, X_IRS)

#7 UBR

X_user_UBR=torch.einsum('bij,bjk->bik',adj_user_BS, X_BS_BR)



#8 B
X_BS_B=X_BS

# 9 BU

X_BS_BU=torch.einsum('bij,bjk->bik',torch.permute(adj_user_BS,(0,2,1)), X_user)

# 10 RU (IRS)

X_IRS_RU=torch.einsum('bij,bjk->bik',torch.permute(adj_user_IRS,(0,2,1)), X_user)

# 11 BRU

X_BS_BRU=torch.einsum('ij,bjk->bik',adj_BS_IRS, X_IRS_RU)

#12 BUR

X_BS_BUR=torch.einsum('bij,bjk->bik',torch.permute(adj_user_BS,(0,2,1)), X_user_UR)


X_BS_f=torch.cat((X_BS_B,X_BS_BR,X_BS_BU,X_BS_BRU,X_BS_BUR),2)

#13 R

X_IRS_R=X_IRS

# 14 RUB

X_IRS_RUB=torch.einsum('bij,bjk->bik',torch.permute(adj_user_IRS,(0,2,1)), X_user_UB)

# 15 RBU

X_IRS_RBU=torch.einsum('bij,bjk->bik',torch.permute(adj_user_IRS,(0,2,1)), X_BS_BU)




X_user_f=torch.cat((X_user_U,X_user_UR,X_user_UB,X_user_URB,X_user_UBR),2)

X_BS_f=torch.cat((X_BS_B,X_BS_BR,X_BS_BU,X_BS_BRU,X_BS_BUR),2)

X_IRS_f=torch.cat((X_IRS_R,X_IRS_RU,X_IRS_RB,X_IRS_RUB,X_IRS_RBU),2)


feature_user=X_user_f.shape[2]

feature_BS=X_BS_f.shape[2]

feature_IRS=X_IRS_f.shape[2]

args["feature_user"]=feature_user

args["feature_BS"]=feature_BS

args["feature_IRS"]=feature_IRS

X_U_F=torch.reshape(X_user_f,[args["samples"],args.get('num_users')*feature_user])

X_B_F=torch.reshape(X_BS_f,[args["samples"],1*feature_BS])

X_IRS_F=torch.reshape(X_IRS_f,[args["samples"],1*feature_IRS])

X_final1=torch.cat((X_U_F,X_B_F,X_IRS_F),1)



angle_ur=torch.reshape(angles_uR,[args["samples"],args.get('num_users')*args.get("IRS_elements")])

angle_ub=torch.reshape(angles_uB,[args["samples"],args.get('num_users')*args.get("BS_antenna")])



dist_ur=torch.reshape(dist_uR,[args["samples"],args.get('num_users')*1])
dist_ub=torch.reshape(dist_uB,[args["samples"],args.get('num_users')*1])

angle_ubr=torch.reshape(angles_uBR,[args["samples"],args.get('num_users')*2*args.get("user_antenna")])

X_final=torch.cat((X_final1,angle_ur,angle_ub,dist_ur,dist_ub,angle_ubr),1)

X_train=X_final[0:args.get("train_s"),:]

#X_train2=X_final_user2[0:args.get("train_s"),:,:,:]



X_valid=X_final[args.get("train_s"):args.get("train_s")+args.get("validation_s"),:]

#X_valid2=X_final_user2[args.get("train_s"):args.get("train_s")+args.get("validation_s"),:,:,:]

X_test=X_final[args.get("train_s")+args.get("validation_s"):args.get("train_s")+args.get("validation_s")+args.get("test_s"),:]

#X_test2=X_final_user2[args.get("train_s")+args.get("validation_s"):args.get("train_s")+args.get("validation_s")+args.get("test_s"),:,:,:]



train = DataLoader(X_train.to(device), batch_size=args["batch"], shuffle=False,drop_last=True)

#train2 = DataLoader(X_train2, batch_size=args["batch"], shuffle=False)

valid = DataLoader(X_valid.to(device), batch_size=args["batch"], shuffle=False,drop_last=True)
#valid2 = DataLoader(X_valid2, batch_size=args["batch"], shuffle=False)

test = DataLoader(X_test.to(device), batch_size=args["batch"], shuffle=False,drop_last=True)

