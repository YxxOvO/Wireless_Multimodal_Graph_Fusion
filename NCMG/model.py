import os
# os.environ["CUDA_VISIBLE_DEVICES"]='8'
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataload import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features1,out_features2):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features1 = out_features1
        
        self.out_features2 = out_features2
        
        if out_features2==1:
            
            
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features1)).to(device)
        
            self.bias = nn.Parameter(torch.FloatTensor(out_features1)).to(device)
            
        else:
            
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features1,out_features2)).to(device)
        
            self.bias = nn.Parameter(torch.FloatTensor(out_features1,out_features2)).to(device)
            
            
            
        #self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        
        
        #w1=torch.einsum('ni,io->no', e1*e2, self.weight)
        #w2=torch.einsum('ni,io->no', w1,e2.transpose(0,1))
        
        if self.out_features2==1:
                               
            output = torch.einsum('bmi,io->bmo', input, self.weight)+self.bias
        else:
            
            output = torch.einsum('bmi,ipo->bmpo', input, self.weight)+self.bias
        #bias=torch.matmul(e1*e2,self.bias)
        
        return output
    

    
class MLPLayer_comp(nn.Module):
    def __init__(self, in_features, out_features1,out_features2):
        super(MLPLayer_comp, self).__init__()
        
        self.in_features = in_features
        self.out_features1 = out_features1
        
        self.out_features2 = out_features2
        
        if out_features2==1:
            
            
            self.weight = nn.Parameter(torch.randn([in_features, out_features1],dtype=torch.cfloat)).to(device)
            
            
        
            self.bias = nn.Parameter(torch.randn([out_features1],dtype=torch.cfloat)).to(device)
            
        else:
            
            self.weight = nn.Parameter(torch.randn([in_features, out_features1,out_features2],dtype=torch.cfloat)).to(device)
        
            self.bias = nn.Parameter(torch.randn([out_features1,out_features2],dtype=torch.cfloat)).to(device)
            
            
            
        #self.register_parameter('bias', None)
        #self.reset_parameters()

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        
        
        #w1=torch.einsum('ni,io->no', e1*e2, self.weight)
        #w2=torch.einsum('ni,io->no', w1,e2.transpose(0,1))
        
        if self.out_features2==1:
            
            output = torch.einsum('bmi,io->bmo', input, self.weight)+self.bias
        else:
            
            output = torch.einsum('bmi,ipo->bmpo', input, self.weight)+self.bias
        #bias=torch.matmul(e1*e2,self.bias)
        
        return output    



class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        
        self.hid=self.n_channels//4
        

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)
        
        
        self.q=nn.Parameter(torch.FloatTensor(self.n_channels, self.n_channels//4)).to(device)

        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        
        

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, N, C = x.size() # batchsize, nodes, channels
        

        f = self.query(x)# 
        g = self.key(x)  # 
        h = self.value(x) #
        
        beta=F.softmax(torch.einsum('bij,bjk->bik', f, g.permute(0,2,1))/(math.sqrt(self.hid)),dim=2)

        
        beta = self.att_drop(beta)
        
        
        aa=h[:,:,:,None]

        b=beta.permute(0,2,1)

        bb=b[:,:,None,:]

        aa=aa.repeat(1,1,1,N)

        bb=bb.repeat(1,1,C,1)

        c=bb*aa

        d=torch.sum(c,1)

        d=d.permute(0,2,1)
        
        
        return self.gamma*d + x


class VisualGNN(nn.Module):
    def __init__(self, args, node_1, feature_1, node_2, feature_2, node_3, feature_3, hidden, dropout):
        super(VisualGNN, self).__init__()
        
        self.layers_1 = nn.Sequential(
            MLPLayer(feature_1, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layers_2 = nn.Sequential(
            MLPLayer(feature_2, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layers_3 = nn.Sequential(
            MLPLayer(feature_3, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layer_mid = Transformer(hidden, num_heads=1)

    def forward(self, x1, x2, x3):
        features1 = self.layers_1(x1)
        features2 = self.layers_2(x2)
        features3 = self.layers_3(x3)
        
        feature = torch.concat([features1, features2, features3], dim=1)
        features = self.layer_mid(feature)
        
        return features


class WirelessGNN(nn.Module):
    def __init__(self, args, node_1, feature_1, node_2, feature_2, node_3, feature_3, hidden, dropout):
        super(WirelessGNN, self).__init__()
        
        # 与原SeHGNN相同的结构
        self.layers_1 = nn.Sequential(
            MLPLayer(feature_1, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layers_2 = nn.Sequential(
            MLPLayer(feature_2, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.RReLU(),
            nn.Dropout(dropout),
        )
        
        self.layers_3 = nn.Sequential(
            MLPLayer(feature_3, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.RReLU(),
            nn.Dropout(dropout),
        )
        
        self.layer_mid = Transformer(hidden, num_heads=1)

    def forward(self, x1, x2, x3):
        features1 = self.layers_1(x1)
        features2 = self.layers_2(x2)
        features3 = self.layers_3(x3)
        
        feature = torch.concat([features1, features2, features3], dim=1)
        features = self.layer_mid(feature)
        
        return features

class CrossStitchUnit(nn.Module):
    def __init__(self, features1, features2):
        super(CrossStitchUnit, self).__init__()
        # Learnable matrix for feature combination
        self.features1 = features1
        self.features2 = features2
        self.cross_stitch = nn.Parameter(
            torch.eye(features1 + features2)  # Initialize as identity matrix
        )

    def forward(self, feature1, feature2):
        # Reshape inputs if needed
        batch_size = feature1.size(0)
        seq_len = feature1.size(1)
        
        # Reshape to 2D for matrix multiplication
        feature1_flat = feature1.reshape(-1, self.features1)  # [batch*seq_len, features1]
        feature2_flat = feature2.reshape(-1, self.features2)  # [batch*seq_len, features2]
        
        # Concatenate features
        combined_features = torch.cat([feature1_flat, feature2_flat], dim=-1)  # [batch*seq_len, features1+features2]
        
        # Apply cross-stitch matrix
        fused_features = torch.matmul(combined_features, self.cross_stitch)  # [batch*seq_len, features1+features2]
        
        # Split and reshape back
        fused1 = fused_features[:, :self.features1].reshape(batch_size, seq_len, self.features1)
        fused2 = fused_features[:, self.features1:].reshape(batch_size, seq_len, self.features2)
        
        return fused1, fused2

class NCMG(nn.Module):
    def __init__(self, args, node_1, feature_1_wireless, feature_1_visual, node_2, 
                 feature_2_wireless, feature_2_visual, node_3, feature_3_wireless, 
                 feature_3_visual, hidden, dropout):
        super(NCMG, self).__init__()
        
        self.args = args
        self.hidden = hidden
        self.node1 = node_1
        self.node2 = node_2
        self.node3 = node_3
        
        # Initialize Visual and Wireless GNNs
        self.visual_gnn = VisualGNN(args, node_1, feature_1_visual, node_2, 
                                  feature_2_visual, node_3, feature_3_visual, hidden, dropout)
        self.wireless_gnn = WirelessGNN(args, node_1, feature_1_wireless, node_2, 
                                      feature_2_wireless, node_3, feature_3_wireless, hidden, dropout)
        
        # Cross-Stitch Units
        self.cross_stitch_initial = CrossStitchUnit(hidden, hidden)
        # self.cross_stitch_gnn1 = CrossStitchUnit(hidden, hidden)
        # self.cross_stitch_gnn2 = CrossStitchUnit(hidden, hidden)
        # self.cross_stitch_transformer_input = CrossStitchUnit(hidden * 2, hidden * 2)
        self.cross_stitch_transformer_output = CrossStitchUnit(hidden * 2, hidden * 2)
        
        self.transformer = Transformer(hidden * 2, num_heads=1)
        
        # beamforming
        
        self.layer_1p=MLPLayer_comp(hidden*2,args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),1)
        #self.layer_norm_1p=nn.LayerNorm([args.get("num_users"),args.get("BS_antenna")*args.get("sub_bands")])
        self.layer_2p=MLPLayer_comp(args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),1)
        #self.layer_norm_2p=nn.LayerNorm([args.get("num_users"),args.get("BS_antenna")*args.get("sub_bands")])
        #self.layer_3p=MLPLayer_comp(args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),1)
        
        
        #phase shift amplitude
        self.layer_1phi=MLPLayer(hidden*2,args.get("IRS_elements"),1)
        self.layer_norm_1phi=nn.LayerNorm([1,self.args.get("IRS_elements")])
        self.layer_2phi=MLPLayer(args.get("IRS_elements"),args.get("IRS_elements"),1)
        self.layer_norm_2phi=nn.LayerNorm([1,self.args.get("IRS_elements")])
        self.layer_3phi=MLPLayer(args.get("IRS_elements"),args.get("IRS_elements"),1)
        
        #bandwidth
        self.layer_1b=MLPLayer(hidden*2,args.get("sub_bands"),1)
        self.layer_norm_1b=nn.LayerNorm([1,args.get("sub_bands")])
        self.layer_2b=MLPLayer(args.get("sub_bands"),args.get("sub_bands"),1)
        self.layer_norm_2b=nn.LayerNorm([1,args.get("sub_bands")])
        self.layer_3b=MLPLayer(args.get("sub_bands"),args.get("sub_bands"),1)

       

        self.prelu = nn.RReLU(0.1,0.4)
        self.dropout = nn.Dropout(dropout)
        
        self.sig=nn.Sigmoid()

        #self.reset_parameters()

    def reset_parameters(self):


        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.final_layer.weight, gain=gain)
        nn.init.zeros_(self.final_layer.bias)

    def forward(self, x1_wireless, x2_wireless, x3_wireless, x1_visual, x2_visual, x3_visual, e):
        # ==========================
        # Step 1: 初始层模态交互
        # ==========================
        # Process through individual GNNs
        visual_features = self.visual_gnn(x1_visual, x2_visual, x3_visual)
        wireless_features = self.wireless_gnn(x1_wireless, x2_wireless, x3_wireless)
        visual_features, wireless_expanded = self.cross_stitch_initial(visual_features, wireless_features)

        # ==========================
        # Step 2: GNN 第一层交互
        # ==========================
        # visual_features, wireless_expanded = self.cross_stitch_gnn1(visual_features, wireless_expanded)

        # ==========================
        # Step 3: GNN 第二层交互
        # ==========================
        # visual_features, wireless_expanded = self.cross_stitch_gnn2(visual_features, wireless_expanded)

        # ==========================
        # Step 4: Transformer 输入层交互
        # ==========================
        fused_input = torch.cat([visual_features, wireless_expanded], dim=2)  # [batch, num_users, hidden * 2]
        # fused_input, _ = self.cross_stitch_transformer_input(fused_input, fused_input)

        # ==========================
        # Step 5: Transformer 处理
        # ==========================
        fused_features = self.transformer(fused_input)  # [batch, num_users, hidden * 2]

        # ==========================
        # Step 6: Transformer 输出层交互
        # ==========================
        fused_features, _ = self.cross_stitch_transformer_output(fused_features, fused_features)

        # ==========================
        # Step 7: 后续处理（如 beamforming）
        # ==========================

        p=fused_features[:,0:self.node1,:]
        
        p0=torch.zeros(p.shape[0],p.shape[1],p.shape[2]).to(device)
        
        
        c=torch.stack((p,p0),dim=3)
        
        p=torch.view_as_complex(c)
        
        
        
        phi=fused_features[:,self.node1:self.node1+self.node2,:]
        
        b=fused_features[:,self.node1+self.node2:self.node1+self.node2+self.node3,:]
        
        p_f=self.layer_2p(self.layer_1p(p))
        
        
        p_fr=torch.reshape(p_f,[self.args.get("batch"),self.args.get("num_users")*self.args.get("BS_antenna")*self.args.get("sub_bands")*args.get("stream")])
        
        denom=torch.sqrt(torch.sum(torch.abs(p_fr).pow(2),dim=1))
        
        denom=denom[:,None]
        
        p_rf=denom.expand(-1,self.args.get("num_users")*self.args.get("BS_antenna")*self.args.get("sub_bands")*args.get("stream"))
        
        p_ff=math.sqrt(self.args.get("P_max"))*torch.div(p_fr,p_rf)
        
        beamforming=torch.reshape(p_ff,[self.args.get("batch"),self.args.get("num_users"),self.args.get("BS_antenna"),args.get("stream"),self.args.get("sub_bands")])
        
        #phase shift and amplitude
        
        phi_h=self.layer_2phi(F.relu(self.layer_norm_1phi(self.layer_1phi(phi))))
        
        
        phi_h1=torch.reshape(phi_h,[self.args.get("batch"),1,self.args.get("IRS_elements"),1])
        
        phi_h1[:,:,:,0]=2*torch.tensor(math.pi)*self.sig(phi_h1[:,:,:,0])
        
        #bandwidth
        b0=F.relu(self.layer_norm_1b(self.layer_1b(b)))
        
        b_f=F.relu(self.layer_norm_2b(self.layer_2b(b0)))
        
        cons=(self.args.get("f_end")-self.args.get("f_start")-self.args.get("b_g")*(self.args.get("sub_bands")-1))
        
        b_fi=self.args.get("b_max")*self.sig(b_f)
        
        b_ff=torch.sum(b_fi[:,0,:],dim=1)
        
        b_ff=b_ff[:,None]
        
        b_ff=b_ff.expand(-1,self.args.get("sub_bands"))
        
        b_final=cons*torch.div(b_fi[:,0,:],b_ff)
        
        
        b_final=self.args.get("b_max")-F.relu(self.args.get("b_max")-b_final)
        
        torch.cuda.empty_cache
        
        return beamforming, phi_h1, b_final[:,None,:] 