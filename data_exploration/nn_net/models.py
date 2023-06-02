import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.init as init
import math
import numpy as np

#from multibox_loss import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FFN(nn.Module):
    def __init__(self, input_size=200, final_size=200,activation=True,dropout=0.2):
        super(FFN, self).__init__()
        self.state_size = input_size
        #self.layer_normal = nn.LayerNorm(input_size)
        self.lr1 = nn.Linear(input_size, final_size)
        self.activation=nn.ReLU()# nn.LeakyReLU() #
        self.dropout = nn.Dropout(dropout)
        self.apply_activation=activation
        self.init_layer(self.lr1)
        #self.batchN=nn.BatchNorm1d(num_features=final_size,affine =True)

    def forward(self, x):
        x = self.lr1(x)
        #adding batch normalization
        #x=self.batchN(x)
        if self.apply_activation:x = self.activation(x)

        return self.dropout(x)

    @staticmethod
    def init_layer(layer):
        nn.init.kaiming_uniform(layer.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)
        #layer.weight.data.fill_(1)

    def init_weight(self):
            self.init_layer(self.fc1)


class simple_nn(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.1):
        super(simple_nn, self).__init__()
        self.ffns=nn.ModuleList()
        #self.dropout = nn.Dropout(dropout)
        self.layer_norm=nn.ModuleList()
        self.layer_normal0 = nn.LayerNorm(input_size*13)
        self.layer_normal1 = nn.LayerNorm(input_size*13*2)
        self.dropout=dropout
        # featur engenereing in params 0,1 | time extracting variable sin params 2,3, |100 combination from column space
        #params=[(input_size*13,input_size*13*2,0,0),(input_size*13*2,100,1,0),(100,output_size,0,0)]#,(13,1,0,0),(1,1,0,0),(1*100,100,0,1),(100,1,0,1)]
        params = [(input_size ,input_size  * 4,1,0),
                  (input_size  * 4, input_size  * 2,1,0),(input_size  * 2,1,0,0)]#,(1,1,0,0),(1*100,100,0,1),(100,1,0,1)]
        loop=0
        self.params=[]
        for param in params:
            loop += 1
            self.params.append(param)
            #if param[3]==1:self.layer_norm.append(nn.LayerNorm(param[0]))
            #else:self.layer_norm.append(nn.LayerNorm(0))

            self.ffns.append(FFN(param[0],param[1], dropout=self.dropout, activation=param[2]))

    def forward(self, x):

        for i in range(len(self.ffns)):
            # dim 0-> batch 1->different time 2-> diffrent variable for a time (batch,13,24)
            x=x.flatten(start_dim=1)
            #if self.params[i][2] != 0: x = self.layer_norm[i](x)
            x = self.ffns[i](x)
        return F.sigmoid(x.flatten()) #, att_weight
class time_combination_nn(nn.Module):
    def __init__(self,dropout=0.2,input_size=0,output_size=0):
        super(time_combination_nn, self).__init__()
        self.ffns_0 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffns_1 = nn.ModuleList()
        self.ffns2=nn.ModuleList()

        #self.attention=simple_attention_block(input_size=188,output_size=1,num_heads=1,dropout=0.2,num_blocks=1)
        #for i in range(13): self.ffns_0.append(FFN(188,10, dropout=dropout, activation=False))

        for i in range(input_size):self.ffns.append(FFN(13,20,dropout=dropout, activation=False))

        for i in range(input_size): self.ffns_1.append(FFN(20, 13, dropout=dropout, activation=True))
        self.ffns2.append(FFN(input_size*13,100,dropout=dropout, activation=True))
        self.ffns2.append(FFN(100, 1, dropout=dropout, activation=False))
        #self.reset_parameters()
        #self.layer_normal0 = nn.LayerNorm(input_size )
        #self.layer_normal1 = nn.LayerNorm(input_size*13)
    def forward(self, x):
         x = x.permute(0, 2, 1)
         output = torch.zeros((x.shape[0], x.shape[1], 13))
         for j in range(len(self.ffns)):
             output[:,j,:]= self.ffns_1[j](self.ffns[j](x[:,j,:]))

         x=output.flatten(start_dim=1)

         for j in range(len(self.ffns2)):
              x=self.ffns2[j](x)
         return torch.sigmoid(x.flatten())


        # init_layer(self.conv2)
        # init_bn(self.bn1)
        # init_bn(self.bn2)


class time_combination_nn_deeper(nn.Module):
    def __init__(self, dropout=0.2, input_size=0, output_size=0):
        super(time_combination_nn_deeper, self).__init__()

        self.ffns = nn.ModuleList()
        for i in range(8):
            if i >= 2:
                pass
            else:
                self.ffns.append(nn.ModuleList())
        # self.ffns[0],self.ffns[1]=nn.ModuleList(),nn.ModuleList()
        for i in range(input_size): self.ffns[0].append(FFN(13, 20, dropout=dropout, activation=True))
        for i in range(input_size): self.ffns[1].append(FFN(20, 13, dropout=dropout, activation=True))
        self.ffns.append(FFN(input_size * 13, 256, dropout=dropout, activation=True))
        self.ffns.append(FFN(256, 64, dropout=dropout, activation=True))
        self.ffns.append(FFN(64, 64, dropout=dropout, activation=True))
        self.ffns.append(FFN(320, 64, dropout=dropout, activation=True))
        self.ffns.append(FFN(64, 16, dropout=dropout, activation=True))
        self.ffns.append(FFN(16, 1, dropout=dropout, activation=False))

    def forward(self, x):

        x = x.permute(0, 2, 1)  # .to(device)
        if torch.cuda.is_available():
            output = torch.zeros((x.shape[0], x.shape[1], 13), device='cuda')  # .to(device)
        else:
            output = torch.zeros((x.shape[0], x.shape[1], 13))
        for j in range(len(self.ffns[0])):
            output[:, j, :] = self.ffns[1][j](self.ffns[0][j](x[:, j, :]))
        output = self.ffns[2](output.flatten(start_dim=1))

        x = self.ffns[4](self.ffns[3](output))
        x = torch.cat((output, x), dim=1)
        x = self.ffns[7](self.ffns[6](self.ffns[5](x)))
        return torch.sigmoid(x.flatten())
class time_combination_nn_with_variable_mixer(nn.Module):
    def __init__(self,dropout=0.2,input_size=0,output_size=0):
        super(time_combination_nn_with_variable_mixer, self).__init__()
        self.ffns_0 = nn.ModuleList()
        self.ffns_01 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffns_10 = nn.ModuleList()
        self.ffns_11 = nn.ModuleList()
        self.ffns_12 = nn.ModuleList()
        self.ffns2=nn.ModuleList()
        self.ffns_20 = nn.ModuleList()
        self.ffns_21 = nn.ModuleList()

        #variable_mixing
        self.ffns_0.append(FFN(input_size,input_size*2, dropout=dropout, activation=True))

        self.ffns_01.append(FFN( input_size * 2,input_size, dropout=dropout, activation=False))
        # seperate time for mixed variable
        self.ffns_20.append(FFN(13, 30, dropout=dropout, activation=False))
        self.ffns_21.append(FFN(30, 30, dropout=dropout, activation=True))
        #self.ffns_21.activation=nn.Sigmoid()
        # self.ffns_0[0].lr1.weight.requires_grad = False
        # self.ffns_0[0].lr1.bias.requires_grad = False
        # self.ffns_01[0].lr1.weight.requires_grad = False
        # self.ffns_01[0].lr1.bias.requires_grad = False


        #time mixing
        # for i in range(input_size):self.ffns.append(FFN(13,20,dropout=dropout, activation=False))
        # for i in range(input_size): self.ffns_1.append(FFN(20, 13, dropout=dropout, activation=True))
        self.ffns_10.append(FFN(13,30,dropout=dropout, activation=False))
        self.ffns_11.append(FFN(30, 13, dropout=dropout, activation=True))
        self.ffns_12.append(FFN(13, 30, dropout=dropout, activation=False))




        #flatten ffn
        self.ffns2.append(FFN(input_size * 30*2 , 100, dropout=dropout, activation=True))  # True:2 F:1 testing
        #self.ffns2.append(FFN(input_size*30*2,100,dropout=dropout, activation=True))#True:2 F:1 testing
        self.ffns2.append(FFN(100, 1, dropout=dropout, activation=False))
        self.layer_normal0 = nn.LayerNorm(31)
        self.layer_normal1 = nn.LayerNorm(input_size*2)
    def forward(self, x):
         #y=x[:]
         #x = self.layer_normal0(x)
         #output2 = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))

         #v mixing and time ixing together
         output2=self.ffns_01[0](self.ffns_0[0](x[:,:,:])) #+x
         output2=self.ffns_21[0](self.ffns_20[0](output2.permute(0,2,1)))
         #output2=F.normalize(output2,dim=2)
         # fragmented fnns

         #x=y[:]
         #output2=torch.zeros(x.shape[0],x.shape[1],x.shape[2])

         #x=torch.zeros(x.shape)
         #x=torch.cat((x,output2),dim=2)

         x = x.permute(0, 2, 1)
         #output = torch.zeros((x.shape[0], x.shape[1], 26))
         # for j in range(len(self.ffns)):
         #     output[:,j,:]= self.ffns_1[j](self.ffns[j](x[:,j,:]))
         #time mixing
         output = self.ffns_11[0](self.ffns_10[0](x[:, :, :]))
         output=self.ffns_12[0](output)# skip connection  problem time varibale ixing not workng for varibale mixed
         x = torch.cat((output, output2), dim=1)
         # varibale metrics mixing
         #output=self.layer_normal1(output.permute(0,2,1))
         #output = self.ffns_21[0](self.ffns_20[0](output))

         #x=x+output # res connection
         #              ,x.std(dim=2).reshape((x.shape[0],188,1)),torch.max(x,dim=2)[0].reshape(x.shape[0],188,1),
         #              torch.min(x,dim=2)[0].reshape(x.shape[0],188,1)),dim=2)
         #x=torch.cat((output.flatten(start_dim=1),output2.flatten(start_dim=1)),dim=1)
         x=x.flatten(start_dim=1)
         #x=self.attention(output)

         #adding a attentionlayer
         #x=self.layer_normal1(x.flatten(start_dim=1))
         for j in range(len(self.ffns2)):
              x=self.ffns2[j](x)
         return torch.sigmoid(x.flatten())

class time_combination_nn2(nn.Module):
    def __init__(self, dropout=0.2, input_size=0, output_size=0):
        super(time_combination_nn2, self).__init__()
        self.ffns_00 = nn.ModuleList()
        self.ffns_01 = nn.ModuleList()
        self.ffns_02 = nn.ModuleList()

        self.ffns_20 = nn.ModuleList()
        self.ffns_21 = nn.ModuleList()
        self.ffns_1 = nn.ModuleList()



        # self.attention=simple_attention_block(input_size=188,output_size=1,num_heads=1,dropout=0.2,num_blocks=1)
        # for i in range(13): self.ffns_0.append(FFN(188,10, dropout=dropout, activation=False))
        for i in range(input_size):
            #temp1=nn.ModuleList()
            #temp2=nn.ModuleList()
            temp=nn.ModuleList()
            #for j in range(13):
                #temp1.append(FFN(2, 2, dropout=0, activation=False))
                #temp2.append(FFN(2, 1, dropout=0, activation=True))
            temp.append(FFN(2, 1, dropout=0, activation=False))
            self.ffns_00.append(temp)

        #time mixing
        self.ffns_01.append(FFN(13, 20, dropout=dropout, activation=False))
        self.ffns_02.append(FFN(20, 13, dropout=dropout, activation=True))


        #self.ffns_20.append(FFN(input_size*2, 20, dropout=dropout, activation=True))
        #self.ffns_21.append(FFN(20, 13, dropout=dropout, activation=False))


        self.ffns_1.append(FFN(input_size * 13, 100, dropout=dropout, activation=True))
        self.ffns_1.append(FFN(100, 1, dropout=dropout, activation=False))

        self.layer_normal1 = nn.LayerNorm(input_size * 2)

    def forward(self, x):
        #x=self.layer_normal1(x)
        #output3 = torch.zeros((x.shape[0],13))
        #for j in range(len(self.ffns_20)):
        #output3=self.ffns_21[0](self.ffns_20[0](x[:,-1,:]))
        #x=torch.cat((x,output2),dim=2)
        x = x.permute(0, 2, 1)
        output = torch.zeros((x.shape[0], int(x.shape[1]/2), 13))
        output2 = torch.zeros((x.shape[0], 13,int(x.shape[1]/2)))
        for j in range(int(x.shape[1]/2)):
            temp=x[:, j*2:j*2+2, :].permute(0,2,1)

            #for k in range(13):
            output2[:,:,j]=self.ffns_00[j][0](temp[:,:,:]).flatten(start_dim=1)
            #output[:, j, :] = self.ffns_1[j](self.ffns[j](torch.squeeze(self.ffns_0[j](temp).permute(0,2,1))))
        output[:, :, :] = self.ffns_02[0](self.ffns_01[0](output2.permute(0,2,1)))
        # x=x+output # res connection
        #              ,x.std(dim=2).reshape((x.shape[0],188,1)),torch.max(x,dim=2)[0].reshape(x.shape[0],188,1),
        #              torch.min(x,dim=2)[0].reshape(x.shape[0],188,1)),dim=2)
        # x=torch.cat((output.flatten(start_dim=1),output2.flatten(start_dim=1)),dim=1)
        #output=torch.cat((output.flatten(start_dim=1),output3),dim=1)
        x = output.flatten(start_dim=1)
        # x=self.attention(output)

        # adding a attentionlayer
        #x = self.layer_normal1(x.flatten(start_dim=1))
        for j in range(len(self.ffns_1)):
            x = self.ffns_1[j](x)
        return F.sigmoid(x.flatten())



class transformer_v2(nn.Module):
    def __init__(self,input_size,output_size,num_blocks,seq_len=13):
        self.num_blocks=num_blocks
        super(transformer_v2, self).__init__()
        self.encoders_blocks = nn.ModuleList()
        input_size_,output_size_=input_size,input_size
        for i in range(self.num_blocks):
            input_size_ = output_size_
            output_size_ = int(output_size_ / 4) * 2
            #print(input_size_, output_size_)
            self.encoders_blocks.append(transformer_encoder_block(input_size=input_size_,output_size=output_size_,num_heads=2,drop_out=0.2))


        self.final_layer=FFN(input_size=output_size_*seq_len, final_size=output_size)
    def forward(self,x):
        x=x.permute(1,0,2)
        for i in range(self.num_blocks):
            #print(x.shape)
            x,att_weights=self.encoders_blocks[i](x)


        x=self.final_layer(x.permute(1, 0, 2).flatten(start_dim=1))
        return F.sigmoid(x.flatten())
if __name__ == "__main__":
    model=transformer_v1(input_size=188,output_size=1,num_blocks=1)
    out=model(torch.ones((13,10,188)))
    h=0



