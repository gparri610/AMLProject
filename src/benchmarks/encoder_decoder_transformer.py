import torch
import torch.nn as nn

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, src_feat_dim, tgt_feat_dim, dim_model, num_heads, num_layers, dropout=0.1):
        super(EncoderDecoderTransformer, self).__init__()
        self.src_projection = nn.Linear(src_feat_dim, dim_model)
        self.tgt_projection = nn.Linear(tgt_feat_dim, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(dim_model, tgt_feat_dim)
        self.tanh = nn.Tanh()

    def forward(self, src, tgt):
        src = self.src_projection(src)
        tgt = self.tgt_projection(tgt)
        
        # Creating a square subsequent mask for the target (decoder)
        # This mask ensures that positions can only attend to earlier positions in the output sequence
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass with masking
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.fc_out(output)
        output_last = output.squeeze(-1)[:,-1].reshape(-1,1)
        return self.tanh(output_last)