# https://github.com/suvadeep-iitb/TransNet


import torch
from torch import nn
import torch.nn.functional as F

import numpy as np



class RelPosEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        inv_freq = 1 / (10000 ** (torch.arange(0, embed_dim, 2.0)))

        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        pos_seq = pos_seq.cuda()

        sinusoid_inp = torch.einsum("i,j->ij", pos_seq, self.inv_freq)
        pos_emb = torch.concat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return torch.tile(pos_emb[:, None, :], [1, bsz, 1])
        else:
            return pos_emb[:, None, :]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()

        # Pytorch initialization strategies are applied imperatively after instantiating the module
        # Should apply recursively to inner modules as well
        self.layer = nn.Sequential(

            # Input dimension needs to be equal to output dimension
            nn.Linear(d_model, d_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )


    # Pytorch automatically disables dropout layers when model.eval() is called
    def forward(self, inp):
        core_out = self.layer(inp)

        output = [core_out + inp]
        return output






class RelativeMultiHeadAttn(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        dropout,
        dropatt,
        r_r_bias=None, # what is this?
        r_w_bias=None, # what is this?
        smooth_pos_emb=True, # what is this?
        untie_pos_emb=True, # what is this?
        clamp_len=-1, # maximum relative distance
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.clamp_len = clamp_len


        # Author's convention: _net == dense layer
        # Input dimension??
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        if self.smooth_pos_emb:

            # Input dimension???
            self.r_net = nn.Linear(d_model, self.n_head * self.d_head, bias=False)
        elif self.untie_pos_emb:

            # Assume defaults for Embedding in tf and pytorch to be the same
            self.pos_emb = nn.Embedding(2*self.clamp_len+1, d_model)


        self.drop_r = nn.Dropout(dropout)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # output transformation
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.scale = 1 / (d_head ** 0.5)

        if r_r_bias is not None and r_w_bias is not None:  # Biases are shared
            self.r_r_bias = r_r_bias
            self.r_w_bias = r_w_bias
        else:
            self.r_r_bias = nn.Parameter(torch.zeros([self.n_head, self.d_head]))
            self.r_w_bias = nn.Parameter(torch.zeros([self.n_head, self.d_head]))


    def _rel_shift(self, x):
        x_size = x.shape

        # Pytorch padding dimensions has order reversed, and each "end" of dimension to pad is adjacent in list, rather than tuple
        # 8 element list for 4 dimension, pads one at start of second dimension
        x = F.pad(x, [0, 0, 0, 0, 1, 0, 0, 0])
        x = torch.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])

        # tf.slice requires all dimensions explicitly stated, unlike normal numpy array slicing
        x = x[1:, ...]
        x = torch.reshape(x, x_size)

        # maybe
        # x = x_padded[1:].view_as(x)

        return x

    def forward(self, inputs):
        
        w, r = inputs
        qlen, rlen, bsz = w.shape[0], r.shape[0], w.shape[1]


        w_heads = self.qkv_net(w)


        if not self.smooth_pos_emb and self.untie_pos_emb:
            r = self.pos_emb(r)
        r_drop = self.drop_r(r)

        if self.smooth_pos_emb:
            r_head_k = self.r_net(r_drop)
        else:
            r_head_k = r_drop


        chunk_size = int(np.ceil(w_heads.shape[-1] / 3))

        w_head_q, w_head_k, w_head_v = torch.split(w_heads, chunk_size, dim=-1)
        w_head_q = w_head_q[-qlen:]


        klen = w_head_k.shape[0]

        # Reshape into head dimensions
        w_head_q = torch.reshape(w_head_q, (qlen, bsz, self.n_head, self.d_head))
        w_head_k = torch.reshape(w_head_k, (klen, bsz, self.n_head, self.d_head))
        w_head_v = torch.reshape(w_head_v, (klen, bsz, self.n_head, self.d_head))

        r_head_k = torch.reshape(r_head_k, (rlen, self.n_head, self.d_head))

        rw_head_q = w_head_q + self.r_w_bias
        rr_head_q = w_head_q + self.r_r_bias

        AC = torch.einsum("ibnd,jbnd->ijbn", rw_head_q, w_head_k)
        BD = torch.einsum("ibnd,jnd->ijbn", rr_head_q, r_head_k)
        BD = self._rel_shift(BD)
        BD = BD[:, :klen, :, :]

        attn_score = AC + BD
        attn_score = attn_score * self.scale

        # softmax along query dimension
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        attn_vec = torch.einsum("ijbn,jbnd->ibnd", attn_prob, w_head_v)
        size_t = attn_vec.shape
        attn_vec = torch.reshape(attn_vec, (size_t[0], size_t[1], self.n_head * self.d_head))

        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        outputs = [w + attn_out, attn_prob, AC, BD]

        return outputs



class TransformerLayer(nn.Module):
    def __init__(
        self,
        n_head,
        d_model,
        d_head,
        d_inner,
        dropout,
        dropatt,
        r_w_bias=None,
        r_r_bias=None,
        smooth_pos_emb=True,
        untie_pos_emb=True,
        clamp_len=-1,
    ):
        super().__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.d_inner = d_inner
        self.dropout = dropout
        self.dropatt = dropatt
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb
        self.clamp_len = clamp_len

        self.xltran_attn = RelativeMultiHeadAttn(
            n_head=self.n_head,
            d_model=self.d_model,
            d_head=self.d_head,
            dropout=self.dropout,
            dropatt=self.dropatt,
            r_w_bias=r_w_bias,
            r_r_bias=r_r_bias,
            smooth_pos_emb=self.smooth_pos_emb,
            untie_pos_emb=self.untie_pos_emb,
            clamp_len=self.clamp_len,
        )
        self.pos_ff = PositionwiseFF(
            d_model=self.d_model,
            d_inner=self.d_inner,
            dropout=self.dropout,
        )

    def forward(self, inputs):
        inp, r = inputs
        attn_outputs = self.xltran_attn([inp, r])
        ff_output = self.pos_ff(attn_outputs[0])

        outputs = [ff_output[0]] + attn_outputs[1:]

        return outputs




class Transformer(nn.Module):
    def __init__(self, n_layer, d_model, n_head, d_head, d_inner, dropout, dropatt, 
                 n_classes, conv_kernel_size, pool_size, clamp_len=-1, 
                 untie_r=False, smooth_pos_emb=True, untie_pos_emb=True, output_attn=False):

        super(Transformer, self).__init__()

        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner

        self.dropout = dropout 
        self.dropatt = dropatt 

        self.n_classes = n_classes

        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size

        self.clamp_len = clamp_len
        self.untie_r = untie_r
        self.smooth_pos_emb = smooth_pos_emb
        self.untie_pos_emb = untie_pos_emb

        self.output_attn = output_attn

        self.conv1 = nn.Conv1d(1, self.d_model, self.conv_kernel_size)
        self.relu1 = nn.ReLU()

        self.pool1 = nn.AvgPool1d(self.pool_size, self.pool_size)

        if self.smooth_pos_emb:
            self.pos_emb = RelPosEmbedding(d_model)
        else:
            assert(self.clamp_len > 0)
            if not self.untie_pos_emb:
                self.pos_emb = nn.Embedding(2*self.clamp_len+1, d_model)
            else:
                self.pos_emb = None

        if not self.untie_r:
            self.r_w_bias = nn.Parameter(torch.zeros([self.n_head, self.d_head]))
            self.r_r_bias = nn.Parameter(torch.zeros([self.n_head, self.d_head]))

        tran_layers = []
        for _ in range(self.n_layer):
            tran_layers.append(
                TransformerLayer(
                    n_head=self.n_head,
                    d_model=self.d_model,
                    d_head=self.d_head,
                    d_inner=self.d_inner,
                    dropout=self.dropout,
                    dropatt=self.dropatt,
                    r_w_bias=None if self.untie_r else self.r_w_bias,
                    r_r_bias=None if self.untie_r else self.r_r_bias,
                    smooth_pos_emb=self.smooth_pos_emb,
                    untie_pos_emb=self.untie_pos_emb,
                    clamp_len=self.clamp_len,
                )
            )

        # Pytorch needs a ModuleList in order to know the connection between parent and submodule
        self.tran_layers = nn.ModuleList(tran_layers)

        self.out_dropout = nn.Dropout(dropout)

        # Input dimension??
        self.fc_output = nn.Linear(d_model, self.n_classes)

    def forward(self, inp):
        # convert the input dimension from [bsz, len] to [bsz, 1, len]
        inp = torch.unsqueeze(inp, axis=1)


        # apply a single layer convolution and then perform pooling to reduce len
        inp = self.conv1(inp)
        inp = self.relu1(inp)


        inp = self.pool1(inp)


        # the rest of the code uses shapes [len, bsz, features] so we transpose 
        # here from shape [bsz, len, dimension] to shape [len, bsz, features]
        inp = torch.permute(inp, dims=(1, 0, 2))

        # torch conv layers use different dimensions than tensorflow
        inp = torch.permute(inp, dims=(2,1,0))


        slen = inp.shape[0]

        pos_seq = torch.arange(slen - 1, -slen, -1.0)

        if self.clamp_len > 0:
            pos_seq = torch.clamp(pos_seq, -self.clamp_len, self.clamp_len)

        if self.smooth_pos_emb:
            pos_emb = self.pos_emb(pos_seq)
        else:
            pos_seq = pos_seq + torch.abs(torch.min(pos_seq))
            pos_emb = pos_seq if self.untie_pos_emb else self.pos_emb(pos_seq)


        core_out = inp
        out_list = []
        for i, layer in enumerate(self.tran_layers):
            all_out = layer([core_out, pos_emb])
            core_out = all_out[0]
            out_list.append(all_out[1:])
        core_out = self.out_dropout(core_out)

        # take the average across the first (len) dimension to get the final representation
        output = torch.mean(core_out, dim=0)

        # get the final scores for all classes
        scores = self.fc_output(output)

        return scores

        if self.output_attn:
            for i in range(len(out_list)):
                for j in range(len(out_list[i])):
                    out_list[i][j] = torch.permute(out_list[i][j], dims=[2, 3, 0, 1])
            return [scores] + out_list
        else:
            return [scores]

