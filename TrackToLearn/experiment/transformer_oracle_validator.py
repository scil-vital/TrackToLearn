import math
import torch

from torch import nn, Tensor


class PositionalEncoding(nn.Module):
    """ From
    https://pytorch.org/tutorials/beginner/transformer_tutorial.htm://pytorch.org/tutorials/beginner/transformer_tutorial.html  # noqa E504
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x


class TransformerOracle(nn.Module):

    def __init__(self, input_size, output_size, n_head, n_layers, lr):
        super(TransformerOracle, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.n_head = n_head
        self.n_layers = n_layers

        self.embedding_size = 32

        self.cls_token = nn.Parameter(torch.randn((3)))

        layer = nn.TransformerEncoderLayer(
            self.embedding_size, n_head, batch_first=True)

        self.embedding = nn.Sequential(
            *(nn.Linear(3, self.embedding_size),
              nn.ReLU()))

        self.pos_encoding = PositionalEncoding(
            self.embedding_size, max_len=(input_size//3) + 1)
        self.bert = nn.TransformerEncoder(layer, self.n_layers)
        self.head = nn.Linear(self.embedding_size, output_size)

        self.sig = nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, threshold=0.01, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "pred_train_loss"
        }

    def forward(self, x):

        N, L, D = x.shape  # Batch size, length of sequence, nb. of dims
        cls_tokens = self.cls_token.repeat(N, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.embedding(x) * math.sqrt(self.embedding_size)

        encoding = self.pos_encoding(x)

        hidden = self.bert(encoding)

        y = self.head(hidden[:, 0])

        y = self.sig(y)

        return y.squeeze(-1)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: dict):

        hyper_parameters = checkpoint["hyper_parameters"]

        input_size = hyper_parameters['input_size']
        output_size = hyper_parameters['output_size']
        lr = hyper_parameters['lr']
        n_head = hyper_parameters['n_head']
        n_layers = hyper_parameters['n_layers']

        model = TransformerOracle(
            input_size, output_size, n_head, n_layers, lr)

        model_weights = checkpoint["state_dict"]

        # update keys by dropping `auto_encoder.`
        for key in list(model_weights):
            model_weights[key] = \
                model_weights.pop(key)

        model.load_state_dict(model_weights)
        model.eval()

        return model

# class TransformerOracle(nn.Module):
#
#     def __init__(self, input_size, output_size, n_head, n_layers, lr):
#         super(TransformerOracle, self).__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.lr = lr
#         self.n_head = n_head
#         self.n_layers = n_layers
#
#         self.embedding_size = 32
#
#         layer = nn.TransformerEncoderLayer(
#             self.embedding_size, n_head, batch_first=True)
#
#         self.embedding = nn.Sequential(
#             *(nn.Linear(3, self.embedding_size),
#               nn.ReLU()))
#
#         self.pos_encoding = PositionalEncoding(
#             self.embedding_size, max_len=input_size//3)
#         self.bert = nn.TransformerEncoder(layer, self.n_layers)
#         self.head = nn.Linear(self.embedding_size, output_size)
#
#         self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.embedding(x) * math.sqrt(self.embedding_size)
#
#         encoding = self.pos_encoding(x)
#
#         hidden = self.bert(encoding)
#
#         pooled = hidden.mean(dim=1)
#
#         y = self.head(pooled)
#
#         y = self.sig(y)
#
#         return y.squeeze(-1)
#
#     @classmethod
#     def load_from_checkpoint(cls, checkpoint: dict):
#
#         hyper_parameters = checkpoint["hyper_parameters"]
#
#         input_size = hyper_parameters['input_size']
#         output_size = hyper_parameters['output_size']
#         lr = hyper_parameters['lr']
#         n_head = hyper_parameters['n_head']
#         n_layers = hyper_parameters['n_layers']
#
#         model = TransformerOracle(
#             input_size, output_size, n_head, n_layers, lr)
#
#         model_weights = checkpoint["state_dict"]
#
#         # update keys by dropping `auto_encoder.`
#         for key in list(model_weights):
#             model_weights[key] = \
#                 model_weights.pop(key)
#
#         model.load_state_dict(model_weights)
#         model.eval()
#
#         return model
