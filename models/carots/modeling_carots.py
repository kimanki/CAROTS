import torch
import torch.nn as nn

from models.carots.encoder import iTransformer_ENC, TimesNet_ENC, LSTM_ENC, GRU_ENC, GATV2_ENC


class CAROTS(nn.Module):
    def __init__(self, cfg):
        super(CAROTS, self).__init__()
        self.cfg = cfg
        self.cfg_data = cfg.DATA
        self.cfg_carots = cfg.CAROTS

        self.encoder = self._init_encoder()
        self.projector = self._init_projector()
        self.causal_discoverer = self._init_causal_discoverer()
        self.positive_augmentor = self._init_positive_augmentor()
        self.negative_augmentor = self._init_negative_augmentor()

    def _init_encoder(self):
        arch = self.cfg_carots.ENCODER_ARCH
        encoder_classes = {
            'lstm': LSTM_ENC,
            'iTransformer': iTransformer_ENC,
            'GATv2': GATV2_ENC,
            'TimesNet': TimesNet_ENC,
            'gru': GRU_ENC,
        }

        if arch not in encoder_classes:
            raise ValueError(f"Unsupported encoder architecture: {arch}")

        encoder = encoder_classes[arch](self.cfg)
        return encoder
    
    def _init_projector(self):
        cfg_projector = self.cfg_carots.PROJECTOR
        projector = nn.Sequential(
            nn.Linear(cfg_projector.INPUT_DIM, cfg_projector.HIDDEN_DIM),
            nn.BatchNorm1d(cfg_projector.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(cfg_projector.HIDDEN_DIM, cfg_projector.OUTPUT_DIM),
            )

        return projector

    def _init_causal_discoverer(self):
        from models.carots.modeling_cuts_plus import CUTS_Plus_Net
        self.cfg_cuts_plus = self.cfg.CUTS_PLUS
        causal_discoverer = CUTS_Plus_Net(self.cfg).cuda()

        return causal_discoverer

    def _init_positive_augmentor(self):
        from models.carots.modeling_positive_augmentor import PositiveAugmentor
        positive_augmentor = PositiveAugmentor(self.cfg).cuda()

        return positive_augmentor
    
    def _init_negative_augmentor(self):
        from models.carots.modeling_negative_augmentor import NegativeAugmentor
        negative_augmentor = NegativeAugmentor(self.cfg).cuda()

        return negative_augmentor

    def forward(self, x, positive_augment=True, negative_augment=True):
        x_all = x
        
        # Apply positive augmentation if enabled
        if positive_augment:
            self.positive_augmentor.set_causal_discoverer(self.causal_discoverer)
            positive_samples = self.positive_augmentor(x_all, self.causal_discoverer.causality_mtx)
            x_all = torch.concat([x_all, positive_samples], dim=0)
        
        # Apply negative augmentation if enabled
        if negative_augment:
            negative_samples = self.negative_augmentor(x_all, self.causal_discoverer.causality_mtx)
            x_all = torch.concat([x_all, negative_samples], dim=0)

        # Encode the input based on the specified architecture
        if self.cfg_carots.ENCODER_ARCH == 'GATv2':
            enc_out = self.encoder(x_all, self.causal_discoverer.causality_mtx)
        elif self.cfg_carots.ENCODER_ARCH in ('lstm', 'iTransformer', 'TimesNet', 'gru'):
            enc_out = self.encoder(x_all)
        else:
            raise ValueError(f"Unsupported encoder architecture: {self.cfg_carots.ENCODER_ARCH}")
        
        # Project the encoded output
        out = self.projector(enc_out)

        return out
