import os
# os.environ["CUDA_VISIBLE_DEVICES"]='8'
import torch
import torch.nn.functional as F
import math
import random
import time
from torch import nn
from dataload import args
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 TensorBoard 日志记录器，使用 args.get() 获取 log_dir
writer = SummaryWriter(args.get("log_dir"))
print(f"训练日志将保存到: {args.get('log_dir')}")

class Trainer(object):
    def __init__(self, model, optimizer, train_loader,val_loader, test_loader,
                  args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        
        self.la=nn.Parameter(torch.FloatTensor(7))
        
        
        nn.init.constant_(self.la, 0.1)
        
        self.lam1=args.get("lam1")
        
        self.lam2=args.get("lam2")
        
        
        #log
        
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)
    
    
    
    
    
    
    def channel_gain(self, f, int_samp, power, phase, bs, batch):
        
        eta1 = self.args.get("eta_1")
        eta2 = self.args.get("eta_2")
        eta3 = self.args.get("eta_3")
        
        c = 3e8
        j = torch.view_as_complex(torch.FloatTensor([0, 1]))
        
        # BS 方向矢量计算
        a_bs1 = j * self.args.get("angles_B") * 2 * torch.tensor(math.pi) * self.args.get("antenna_space") / c
        f_bs = f[:,:,:,None].repeat(1, 1, 1, self.args.get("BS_antenna"))
        a_bs1 = a_bs1[None, None, None, :].repeat(batch, self.args.get("sub_bands"), int_samp, 1)
        a_bs = (1/math.sqrt(self.args.get("BS_antenna"))) * torch.exp(f_bs * a_bs1)  # B S INT_SAMP N_t
        
        # IRS 方向矢量计算
        a_irs = j * self.args.get("angles_R") * 2 * torch.tensor(math.pi) * self.args.get("IRS_space") / c
        f_irs = f[:,:,:,None].repeat(1, 1, 1, self.args.get("IRS_elements"))
        a_irs1 = a_irs[None, None, None, :].repeat(batch, self.args.get("sub_bands"), int_samp, 1)
        a_irs = (1/self.args.get("IRS_elements")) * torch.exp(f_irs * a_irs1)  # B S INT_SAMP L^2
        
        # BS-IRS 信道计算
        f_br = f[:,:,:,None,None].repeat(1, 1, 1, self.args.get("IRS_elements"), self.args.get("BS_antenna"))
        k_abs = torch.exp(eta1 + eta2 * f_br) + eta3
        f_inv = torch.div(1, f_br)
        alpha_br = torch.exp(-0.5 * k_abs * self.args.get("dist_br")) * (f_inv * (c/(4 * math.pi * self.args.get("dist_br"))))
        
        delay = random.uniform(0, 30) * 1e-9
        comp = -2 * torch.tensor(math.pi) * j * delay
        outer_br = torch.einsum('bsij,bsik->bsijk', a_irs, torch.conj(a_bs)) * torch.exp(comp * f_br)
        h_rb = alpha_br * outer_br  # B S INT_SAMP N_t L
        
        # 接收端矩阵 G_r 计算
        j = torch.view_as_complex(torch.FloatTensor([0, 1]))
        gr = torch.exp(j * phase[:, 0, :, 0])
        b = torch.eye(gr.size(1)).to(device)
        c1 = gr.unsqueeze(2).expand(*gr.size(), gr.size(1))
        G_r = c1 * b  # B L^2 L^2
        
        # 使用全零占位，确保 h_u_all 为复数类型，与 power 类型一致
        h_u_all = torch.ones(
            batch,
            self.args.get("sub_bands"),
            int_samp,
            self.args.get("num_users"),
            self.args.get("user_antenna"),
            self.args.get("BS_antenna"),
            dtype=power.dtype,
            device=power.device
        )
        
        # 后续速率计算部分
        noise = self.args.get("noise_pow") * bs
        noise = noise[:, :, :, :, None, None].repeat(1, 1, 1, 1, self.args.get("user_antenna"), self.args.get("user_antenna"))
        
        power = torch.permute(power, (0, 4, 1, 2, 3))
        rate_s_u1 = torch.einsum('bsiurt,bsutm->bsiurm', h_u_all, power)
        rate_s_u = torch.einsum('bsiurm,bsiump->bsiurp',
                                rate_s_u1,
                                torch.conj(torch.permute(rate_s_u1, (0, 1, 2, 3, 5, 4))))
        
        # 修改身份矩阵，转为复数类型，与 rate_s_u 保持一致
        I = torch.eye(self.args.get("num_users"), dtype=rate_s_u.dtype, device=device)
        I = I[None, None, None, :, :].repeat(batch, self.args.get("sub_bands"), int_samp, 1, 1)
        
        nu = torch.einsum('bsiku,bsiurt->bsikrt', I, rate_s_u)
        de = noise + torch.einsum('bsiku,bsiurt->bsikrt', (1 - I), rate_s_u)
        
        I_r = torch.eye(self.args.get("user_antenna"), dtype=rate_s_u.dtype, device=device)
        I_r = I_r[None, None, None, None, :, :].repeat(batch, self.args.get("sub_bands"), int_samp, self.args.get("num_users"), 1, 1) + 0 * j
        
        rate_f = torch.log2(torch.linalg.det(I_r + torch.einsum('bsiurt,bsiutp->bsiurp', nu, torch.linalg.pinv(de + 1e-10))).real)
        
        # 清理中间变量
        del h_u_all, G_r, I, c1, noise, rate_s_u, rate_s_u1, nu, de, I_r
        torch.cuda.empty_cache()
        
        return rate_f


        
        
            
            #for uu in range(self.args.get("num_users")):
                
             #   if uu!=u:
                    
        #denum=denum+torch.sum(torch.abs(torch.einsum('j,bj->b',h_u_all[uuu,:],power[bbb,:uuu,:,s])).pow(2))
        
        #denum=denum+torch.sum(torch.abs(torch.einsum('j,bj->b',h_u_all[uuu,:],power[bbb,uuu+1:,:,s])).pow(2))
                
            
        #rate_s_u=bs*torch.log2(torch.div(num,denum))
            
            
        
        
    
    
    def optimization_problem(self, power, phase, band):
        
        #A1 = torch.einsum('ij,bjk->bik', self.args.get("adj_rrh_user"), prb)
        #A2 = torch.einsum('ij,bjk->bik', self.args.get("adj_rrh_user"), power)
        
        batch = power.size()[0]
        
        fs = self.args.get("f_start") * torch.ones(batch, self.args.get("sub_bands")).to(device)
        int_samp = self.args.get("int_samp")
        
        bs = band[:, 0, :]
        
        bs1 = bs[:, :, None]
        bs1 = bs1.repeat(1, 1, int_samp)
        
        bs2 = bs[:, :, None]
        bs2 = bs2.repeat(1, 1, self.args.get("num_users"))
        
        bs3 = bs[:, :, None, None]
        bs3 = bs3.repeat(1, 1, int_samp, self.args.get("num_users"))
        
        start = torch.ones(self.args.get("sub_bands"), self.args.get("sub_bands")).to(device)
        start = start - torch.triu(start)
        
        bsg = bs + self.args.get("b_g")
        f_st = fs + torch.einsum('ij,bj->bi', start, bsg)
        f_st = f_st[:, :, None]
        f_st = f_st.repeat(1, 1, int_samp)
        
        ran = torch.tensor(range(int_samp)).to(device)
        ran = ran[None, None, :].repeat(batch, self.args.get("sub_bands"), 1)
        
        ff = f_st + (bs1 / (int_samp - 1)) * ran
        
        # 调用修改后的 channel_gain（注意参数已调整，不再传入dist_ur, dist_ub, ang_ur, ang_ub, ang_uBR）
        r = self.channel_gain(ff, int_samp, power, phase, bs3, batch)
        
        r_s_u = bs2 * torch.sum(r, dim=2) * (1 / int_samp)
        
        cons_rate = F.relu(self.args.get("r_u_min") - torch.sum(r_s_u, dim=1))
        
        cons = (self.args.get("f_end") - self.args.get("f_start") - 
                self.args.get("b_g") * (self.args.get("sub_bands") - 1))
        cons_band = F.relu(cons - torch.sum(band[:, 0, :], dim=1))
        
        loss = (-torch.sum(r_s_u) + self.lam1 * torch.sum(cons_rate)) / (batch * 1e10)
        
        r_s_u = r_s_u.detach()
        band = band.detach()
        
        self.lam1 = F.relu(self.lam1 + torch.sum(self.args.get("r_u_min") - torch.sum(r_s_u, dim=1)) / batch)
        
        # with open('lam1.txt', 'a') as f:
        #     f.write("{}".format(self.lam1))
        #     f.write('\n')
        
        del bs1, bs2, bs3, ff, r, ran, f_st, start
        torch.cuda.empty_cache()
        
        return loss, r_s_u, torch.sum(r_s_u) / batch

        
        
    
    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x_visual) in enumerate(val_dataloader):
                #data = data[..., :self.args.get('input_dim')]
                #label = target[..., :self.args.get('output_dim')]
                batch=x_visual.size()[0]
                
                user=x_visual.size()[1]
                
                
                # 解析视觉特征
                x_user_visual = x_visual[:,0:self.args.get("num_users")*self.args.get("feature_user_visual")]
                x_user_visual = torch.reshape(x_user_visual,[batch,self.args.get("num_users"),self.args.get("feature_user_visual")])
                
                x_bs_visual = x_visual[:,self.args.get("num_users")*self.args.get("feature_user_visual"):self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual")]
                x_bs_visual = torch.reshape(x_bs_visual,[batch,1,self.args.get("feature_BS_visual")])
                
                x_irs_visual = x_visual[:,self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual"):self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual")+self.args.get("feature_IRS_visual")]
                x_irs_visual = torch.reshape(x_irs_visual,[batch,1,self.args.get("feature_IRS_visual")])                
                
                
                # 调用新模型
                power, phase, band = self.model(
                    # x_user, x_bs, x_irs,  # 无线特征
                    x_user_visual, x_bs_visual, x_irs_visual,  # 视觉特征
                    epoch
                )
                
                
                loss,rate,sum_rate=self.optimization_problem(power,phase,band)
                
                   
                #if self.args.get('real_value'):
                    #label = self.scaler.inverse_transform(label)
                #loss = self.loss(output.to(device), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(sum_rate):
                    total_val_loss += sum_rate
        val_loss = total_val_loss / len(val_dataloader)
        print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, sum_rate/1e9))
        # with open('loss_records.txt', 'a') as f:
        #     f.write('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch,sum_rate/1e9) + '\n')
        
        #with open('STAR-IRS-MISO-THz.txt', 'a') as f:
         #   f.write('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
         #   f.write('\n\n')
        
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_sum=0
        cr_t=0
        cb_t=0
        for batch_idx, (x_visual) in enumerate(self.train_loader):
            #data = data[..., :self.args.get('input_dim')]
            #label = target[..., :self.args.get('output_dim')]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            batch=x_visual.size()[0]
                
            user=x_visual.size()[1]
                
            # 解析视觉特征
            x_user_visual = x_visual[:,0:self.args.get("num_users")*self.args.get("feature_user_visual")]
            x_user_visual = torch.reshape(x_user_visual,[batch,self.args.get("num_users"),self.args.get("feature_user_visual")])
                
            x_bs_visual = x_visual[:,self.args.get("num_users")*self.args.get("feature_user_visual"):self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual")]
            x_bs_visual = torch.reshape(x_bs_visual,[batch,1,self.args.get("feature_BS_visual")])
                
            x_irs_visual = x_visual[:,self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual"):self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual")+self.args.get("feature_IRS_visual")]
            x_irs_visual = torch.reshape(x_irs_visual,[batch,1,self.args.get("feature_IRS_visual")])             
                
                
            # 调用新模型
            power, phase, band = self.model(
                    # x_user, x_bs, x_irs,  # 无线特征
                    x_user_visual, x_bs_visual, x_irs_visual,  # 视觉特征
                    epoch
                )
                
                
            loss,rate,sum_rate=self.optimization_problem(power,phase,band)
                

            loss.backward()

            # add max grad clipping
            
            self.optimizer.step()
            total_loss += loss.item()
            
            total_sum=total_sum+sum_rate
            
            # 聚合每个用户的总速率
            total_user_rate = rate.sum(dim=0)  # 对第一维（子频带维度）求和，得到每个用户的总速率
            total_user_rate = total_user_rate.sum(dim=0)
            # 计算 Jain's Fairness Index
            fairness_index = (total_user_rate.sum() ** 2) / (len(total_user_rate) * (total_user_rate ** 2).sum())
            fairness_index = fairness_index.item()  # 转为标量值            
            
            

            if batch_idx % self.args.get('log_step') == 0:
                print('Train Epoch {}: {}/{} Loss: {:.6f}, Fairness Index: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, sum_rate / 1e9, fairness_index))
                # 写入 TensorBoard
                writer.add_scalar("Training/sum_rate", sum_rate / 1e9, epoch * self.train_per_epoch + batch_idx)
                writer.add_scalar("Training/fairness_index", fairness_index, epoch * self.train_per_epoch + batch_idx)

                # 打印每个用户的总速率
                for user_id, user_rate in enumerate(total_user_rate):
                    print(f"User {user_id + 1} Total Rate: {user_rate.item() / 1e9:.6f} Gbps")
                    writer.add_scalar(f"Training/User_{user_id + 1}_rate", user_rate.item() / 1e9, epoch * self.train_per_epoch + batch_idx)
                # 将损失和速率记录到文件
                # with open('loss_records.txt', 'a') as f:
                #     f.write('Train Epoch {}: {}/{} Loss: {:.6f}, Fairness Index: {:.6f}\n'.format(
                #         epoch, batch_idx, self.train_per_epoch, sum_rate / 1e9, fairness_index))

        train_epoch_loss = total_loss / self.train_per_epoch
        sum_rate_t = total_sum / self.train_per_epoch / 1e9

        # 每个 epoch 的总结
        print('**********Train Epoch {}: Averaged Loss: {:.6f}, Avg Sum Rate: {:.6f} Gbps, Fairness Index: {:.6f}'.format(
            epoch, train_epoch_loss, sum_rate_t, fairness_index))
        # 写入 TensorBoard
        writer.add_scalar("Epoch/Average_Loss", train_epoch_loss, epoch)
        writer.add_scalar("Epoch/Average_Sum_Rate", sum_rate_t, epoch)
        writer.add_scalar("Epoch/Average_Fairness_Index", fairness_index, epoch)
        # 将平均损失和速率记录到文件
        # with open('loss_records.txt', 'a') as f:
        #     f.write('**********Train Epoch {}: Averaged Loss: {:.6f}, Avg Sum Rate: {:.6f} Gbps, Fairness Index: {:.6f}\n'.format(
        #         epoch, train_epoch_loss, sum_rate_t, fairness_index))

        # with open('rate.txt', 'a') as f:
        #     f.write("{}".format(sum_rate_t))
        #     f.write('\n')
       
        #learning rate decay
        if self.args.get('lr_decay'):
            self.lr_scheduler.step()
        return train_epoch_loss
        

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.get('epochs') + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            #if train_epoch_loss > 1e6:
            #    print('Gradient explosion detected. Ending...')
            #    break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.get('early_stop'):
                if not_improved_count == self.args.get('early_stop_patience'):
                    print("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.get('early_stop_patience')))
                    break
            # save the best state
            

        training_time = time.time() - start_time
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        writer.close()
        #with open('MHGNN_IRS_MIMO-THz_training_time.txt', 'a') as f:
        #    f.write("{}".format(training_time))
        #    f.write('\n')

        #save the best model to file
        

        #test
        #self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        #y1,y2=self.test(self.model, self.args, self.test_loader,  self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    
    def test(self):
        
        self.model.eval()
        power_out = []
        phase_out = []
        band_out=[]
        rate_out=[]
        total_sum=[]
        with torch.no_grad():
            for batch_idx, (x_visual) in enumerate(self.test_loader):
                batch=x_visual.size()[0]
                
                user=x_visual.size()[1]
                
                # 解析视觉特征
                x_user_visual = x_visual[:,0:self.args.get("num_users")*self.args.get("feature_user_visual")]
                x_user_visual = torch.reshape(x_user_visual,[batch,self.args.get("num_users"),self.args.get("feature_user_visual")])
                
                x_bs_visual = x_visual[:,self.args.get("num_users")*self.args.get("feature_user_visual"):self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual")]
                x_bs_visual = torch.reshape(x_bs_visual,[batch,1,self.args.get("feature_BS_visual")])
                
                x_irs_visual = x_visual[:,self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual"):self.args.get("num_users")*self.args.get("feature_user_visual")+self.args.get("feature_BS_visual")+self.args.get("feature_IRS_visual")]
                x_irs_visual = torch.reshape(x_irs_visual,[batch,1,self.args.get("feature_IRS_visual")])               
                
                
                # 调用新模型
                power, phase, band = self.model(
                    # x_user, x_bs, x_irs,  # 无线特征
                    x_user_visual, x_bs_visual, x_irs_visual,  # 视觉特征
                    100
                )
                
                loss,rate,sum_rate=self.optimization_problem(power,phase,band)
                
                
                rate_out.append(rate)
                power_out.append(power)
                
                phase_out.append(phase)
                
                band_out.append(band)
                total_sum.append(sum_rate)
                
                
                
        #y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        power_out = torch.cat(power_out, dim=0)
        phase_out = torch.cat(phase_out, dim=0)
        band_out=torch.cat(band_out, dim=0)
        rate_out=torch.cat(rate_out,dim=0)
        #total_sum=torch.cat(total_sum)
        #if not args.get('real_value'):
        #    y_pred = torch.cat(y_pred, dim=0)
        #else:
           # y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        #np.save('./{}_true.npy'.format(args.get('dataset')), y_true.cpu().numpy())
        #np.save('./{}_pred.npy'.format(args.get('dataset')), y_pred.cpu().numpy())
        #for t in range(y_true.shape[1]):
        #    mae, rmse, mape, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
        #                                        args.get('mae_thresh'), args.get('mape_thresh'))
        #    logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        #        t + 1, mae, rmse, mape*100))
        #mae, rmse, mape, _ = All_Metrics(y_pred, y_true, args.get('mae_thresh'), args.get('mape_thresh'))
        #logger.info("Average Horizon, MAE: {:.4f}, MSE: {:.4f}, MAPE: {:.4f}%".format(
        #            mae, rmse, mape*100))
        return total_sum,rate_out,power_out,phase_out,band_out