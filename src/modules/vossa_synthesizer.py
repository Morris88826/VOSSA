# Adapted from: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any, Optional
from dataclasses import asdict
import wandb
import pytorch_lightning as pl
import torch
from torchmetrics import MinMetric, MeanMetric
from src.modules import PasrWithQuantModule, MimiVocoderModule
from src.models import MimiDecoderWithTVTimbreModel
from src.components.speaker_module import SpeakerModule
from src.components.mel_synthesizer import MelSynthesizer, PostNet
from src.components.spectrogram import StreamingLogMelSpectrogram
from src.components.tv_timbre import StreamingGlobalOnlyTVTimbre
from src.components.mel_discriminator import MultiScaleMelDiscriminator, CharbonnierLoss, lsgan_d_loss, lsgan_g_loss, FeatureMatchingLoss
from src.components.losses import LossPED, SpeakerConsistencyLoss
from src.components.contrastive_head import ContrastiveHead
from src.components.data_classes import (
    FVQuantizerArgs, SeanetArgs, TransformerArgs, 
    FrameSynthesizerArgs, 
    Lambdas,
    PostNetArgs, VocoderFeatureArgs
)



class VOSSASynthesizer(pl.LightningModule):
    """Example of LightningModule for MNIST classification.
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        seanet_args: SeanetArgs = SeanetArgs(),
        transformer_args: TransformerArgs = TransformerArgs(),
        quantizer_args: FVQuantizerArgs = FVQuantizerArgs(),
        sample_rate: int = 16000,
        frame_rate: int = 50,
        num_codes: int = 200,
        encoder_dim: int = 512,
        postnet_args: PostNetArgs = PostNetArgs(),
        frame_synthesizer_args: FrameSynthesizerArgs = FrameSynthesizerArgs(),
        vocoder_feature_args: VocoderFeatureArgs = VocoderFeatureArgs(),
        lambdas: Lambdas = Lambdas(),
        learning_rate: float = 0.0001,
        lr_betas: list = [0.8, 0.99],
        lr_sch_gamma: float = 0.999996,
        encoder_path: str = None,
        vocoder_path: str = None,
        gan_decoder_path: Optional[str] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # Multiple optimizers need manual optimization
        self.automatic_optimization = False
        self.mel_feature_extractor = StreamingLogMelSpectrogram(**asdict(vocoder_feature_args))

        self._build_encoder(encoder_path, seanet_args, transformer_args, quantizer_args, sample_rate, frame_rate, num_codes, encoder_dim)
        self._build_speaker_encoder(encoder_dim)
        self._build_tvtimbre_encoder(frame_synthesizer_args)
        self._build_decoder(frame_synthesizer_args, postnet_args)
        self._build_vocoder(vocoder_path)
        self._build_discriminators()

        self._build_losses(lambdas)

    def _build_losses(self, lambdas: Lambdas) -> None:
        """Initialize all loss functions."""
        self.lambdas = asdict(lambdas)

        # Mel and GAN loss function
        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.feature_matching = FeatureMatchingLoss()
        self.loss_ped = LossPED()
        self.contrastive_loss_fn = ContrastiveHead(
            in_dim=512, proj_dim=256, temperature=0.05
        )
        self.speaker_loss = SpeakerConsistencyLoss()

        # for averaging loss across batches
        loss_metric = MeanMetric()
        self.val_mel_loss = loss_metric.clone()
        self.val_f0_loss = loss_metric.clone()
        self.test_loss = loss_metric.clone()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()
        print("Loss functions: ", self.lambdas)

    def _build_encoder(
        self,
        encoder_path: str,
        seanet_args: SeanetArgs,
        transformer_args: TransformerArgs,
        quantizer_args: FVQuantizerArgs,
        sample_rate: int,
        frame_rate: int,
        num_codes: int,
        encoder_dim: int,
    ) -> None:
        """Initialize and freeze content encoder."""
        if encoder_path is None:
            self.encoder = PasrWithQuantModule(
                seanet_args=seanet_args,
                transformer_args=transformer_args,
                quantizer_args=quantizer_args,
                sample_rate=sample_rate,
                frame_rate=frame_rate,
                num_codes=num_codes,
                encoder_dim=encoder_dim,
            )
        else:
            self.encoder = PasrWithQuantModule.load_from_checkpoint(
                encoder_path, map_location=self.device
            )

        # Always freeze the content encoder
        self.encoder.freeze()
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

    def _build_speaker_encoder(self, encoder_dim: int) -> None:
        """Initialize speaker encoder with EMA copy."""
        self.n_layers = 5
        self.speaker_encoder = SpeakerModule(
            spk_map_dim=encoder_dim,
            n_layers=self.n_layers,
            local_hidden_dim=encoder_dim,
            global_hidden_dim=encoder_dim,
            use_global_att=True,
        )
    
    def _build_tvtimbre_encoder(self, frame_synthesizer_args: FrameSynthesizerArgs):
        """Initialize TVTimbre encoder"""
        self.tvtimbre_encoder = StreamingGlobalOnlyTVTimbre(
            content_dim=frame_synthesizer_args.input_dimension,
            g_dim=frame_synthesizer_args.timbre_dimension,
            d_out=frame_synthesizer_args.timbre_cond_dim,
        )

    def _build_decoder(self, frame_synthesizer_args, postnet_args):
        decoder_transformer = MelSynthesizer(
            device=self.device, **asdict(frame_synthesizer_args)
        )
        decoder = PostNet(**asdict(postnet_args))
        self.decoder = MimiDecoderWithTVTimbreModel(
            decoder,
            frame_rate=self.hparams.frame_rate,
            sample_rate=self.hparams.sample_rate,
            channels=1,
            encoder_dim=self.hparams.encoder_dim,
            causal = True,
            decoder_transformer=decoder_transformer,
        )
    
    def _build_vocoder(self, vocoder_path):
        self.vocoder = MimiVocoderModule.load_from_checkpoint(
            vocoder_path,
            map_location=self.device
        )
        self.vocoder.freeze()
        for param in self.vocoder.parameters():
            param.requires_grad = False
        self.vocoder.eval()
    
    def _build_discriminators(self):
        ## Discriminators
        self.discriminator = MultiScaleMelDiscriminator(in_ch=1, base_ch=64, n_layers=4, n_scales=3)

    def configure_optimizers(self):
        gen_params = (
            list(self.speaker_encoder.parameters())
            + list(self.tvtimbre_encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.contrastive_loss_fn.parameters())
        )
        opt_g = torch.optim.AdamW(gen_params, self.hparams.learning_rate, betas=self.hparams.lr_betas)
        opt_d = torch.optim.AdamW(self.discriminator.parameters(), self.hparams.learning_rate, betas=self.hparams.lr_betas)

        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.hparams.lr_sch_gamma)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.hparams.lr_sch_gamma)
        return [opt_g, opt_d], [sch_g, sch_d]

    def forward(self, x: torch.Tensor, target_x: torch.Tensor, return_type: str = "wav") -> torch.Tensor:
        _, z_embed, _ = self.encoder(x)
        global_spkr_embed, local_spkr_embed = self.get_speaker_embeddings(target_x)
        mel = self.decode(z_embed, global_spkr_embed, local_spkr_embed, return_coarse=False)

        if return_type == "wav":
            return self.vocoder(mel)
        return mel

    def on_train_start(self):
        self.val_loss_best.reset()

    def decode(self, content, global_spkr_embed: torch.Tensor, local_spkr_embed: torch.Tensor, return_coarse: bool = False):
        tvtimbre = self.tvtimbre_encoder(content, global_spkr_embed)
        (mel_fine, mel_coarse), _ = self.decoder(content, tvtimbre, output_type="mel")
        if return_coarse:
            return mel_fine, mel_coarse
        return mel_fine
    
    def model_step(self, batch: Any):
        y, (y_a1, y_a2, y_b1, y_b2), tokens, pitch, energy, (src_subj_id, tgt_subj_id, _, _) = batch
        y     = y.unsqueeze(1)
        y_a1  = y_a1.unsqueeze(1)
        y_a2  = y_a2.unsqueeze(1)
        y_b1  = y_b1.unsqueeze(1)
        y_b2  = y_b2.unsqueeze(1)

        content = self.get_content_features(y)
        g_spk_b1, l_spk_b1 = self.get_speaker_embeddings(y_b1)
        mel_y_vc_fine = self.decode(content, g_spk_b1, l_spk_b1, return_coarse=False)

        g_spk_a1, l_spk_a1 = self.get_speaker_embeddings(y_a1)
        mel_y_recon_fine, mel_y_recon_coarse = self.decode(content, g_spk_a1, l_spk_a1, return_coarse=True)

        
        y_vc = self.vocoder(mel_y_vc_fine)
        y_recon = self.vocoder(mel_y_recon_fine)

        g_spk_a2, _ = self.get_speaker_embeddings(y_a2)
        g_spk_b2, _ = self.get_speaker_embeddings(y_b2)
        g_spk_a1_recon, _ = self.get_speaker_embeddings(y_recon)
        g_spk_b1_recon, _ = self.get_speaker_embeddings(y_vc)        

        mel_y = self.mel_feature_extractor(y)

        speaker_embs = dict(
            g_a1=g_spk_a1,
            g_a2=g_spk_a2,
            g_b1=g_spk_b1,
            g_b2=g_spk_b2,
            g_a1_recon=g_spk_a1_recon,
            g_b1_recon=g_spk_b1_recon,
        )

        return (y_recon, y_vc, y, y_b1), (mel_y_recon_fine, mel_y_recon_coarse, mel_y), speaker_embs, (src_subj_id, tgt_subj_id)

    @torch.no_grad()
    def get_content_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract content features from audio using frozen encoder."""
        _, z_embed, _ = self.encoder(x)
        return z_embed
    
    def get_speaker_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker features from audio using speaker encoder."""
        with torch.no_grad():
            spk_feat = self.encoder.net.get_speaker_feature(x)
        return self.speaker_encoder(spk_feat)

    # ==================== Loss Computation ====================

    def _spk_consistency_losses(self, speaker_embs) -> dict:
        """Speaker consistency losses."""
        g_a1 = speaker_embs["g_a1"]
        g_a2 = speaker_embs["g_a2"]
        g_b1 = speaker_embs["g_b1"]
        g_b2 = speaker_embs["g_b2"]

        g_a = (g_a1 + g_a2) / 2.0
        g_b = (g_b1 + g_b2) / 2.0
        g_a1_recon = speaker_embs["g_a1_recon"]
        g_b1_recon = speaker_embs["g_b1_recon"]

        return {
            "speaker_recon_loss": self.speaker_loss(g_a1_recon, g_a),
            "speaker_vc_loss":    self.speaker_loss(g_b1_recon, g_b),
        }

    def _contrastive_loss(self, speaker_embs, src_subj_id, tgt_subj_id) -> dict:
        """Contrastive learning loss."""
        g_a1 = speaker_embs["g_a1"]
        g_a2 = speaker_embs["g_a2"]
        g_b1 = speaker_embs["g_b1"]
        g_b2 = speaker_embs["g_b2"]

        z1 = torch.cat([g_a1, g_b1], dim=0)
        z2 = torch.cat([g_a2, g_b2], dim=0)
        speaker_ids = torch.cat([src_subj_id, tgt_subj_id], dim=0)
        return {"contrastive_loss": self.contrastive_loss_fn(z1, z2, speaker_ids)}

    def training_step(self, batch: Any, batch_idx: int):
        # Retrieve optimizers and lr schedulers
        opt_g, opt_d = self.optimizers()
        

        # Forward pass
        (y_recon, y_vc, y, y_b1), (mel_y_recon_fine, mel_y_recon_coarse, mel_y), speaker_embs, (src_subj_id, tgt_subj_id) = self.model_step(batch)

        #### Discriminator Update ####
        opt_d.zero_grad()
        d_losses = {}

        fake_scores_D, _ = self.discriminator(mel_y_recon_fine.detach().unsqueeze(1))
        real_scores_D, _ = self.discriminator(mel_y.unsqueeze(1))
        loss_D = lsgan_d_loss(real_scores_D, fake_scores_D)   
        d_losses['adv_Dx_loss'] = loss_D

        # Manually backprop for discriminator
        self.manual_backward(loss_D)
        opt_d.step()


        #### Generator Update ####
        opt_g.zero_grad()
        g_losses = {}
        
        rec_post = self.charbonnier(mel_y_recon_fine, mel_y)
        rec_pre  = self.charbonnier(mel_y_recon_coarse,  mel_y)
        g_losses['mel_fine_loss'] = rec_post
        g_losses['mel_coarse_loss'] = rec_pre

        fake_scores_G, fake_feats = self.discriminator(mel_y_recon_fine.unsqueeze(1))     # list over scales
        _, real_feats = self.discriminator(mel_y.unsqueeze(1))

        # LSGAN generator and feature matching
        adv_G = lsgan_g_loss(fake_scores_G)
        fm = self.feature_matching(real_feats, fake_feats)
        g_losses['adv_Gx_loss'] = adv_G
        g_losses['adv_Gx_feat_loss'] = fm

        g_losses.update(self._spk_consistency_losses(speaker_embs))
        g_losses.update(self._contrastive_loss(speaker_embs, src_subj_id, tgt_subj_id))
        
        gen_loss = (
            (self.lambdas["mel_fine_loss"] * g_losses["mel_fine_loss"]) + (self.lambdas["mel_coarse_loss"] * g_losses["mel_coarse_loss"])
            + (self.lambdas["adv_Gx_loss"] * g_losses["adv_Gx_loss"]) + (self.lambdas['adv_Gx_feat_loss'] * g_losses["adv_Gx_feat_loss"])
            + (self.lambdas["speaker_recon_loss"] * g_losses["speaker_recon_loss"]) + (self.lambdas["speaker_vc_loss"] * g_losses["speaker_vc_loss"])
            + (self.lambdas["contrastive_loss"] * g_losses["contrastive_loss"])
        )
        g_losses["loss"] = gen_loss
        
        # Manually backprop for generator
        self.manual_backward(gen_loss)
        opt_g.step()

        # Log metrics
        log_dict = {
            "train/total/loss": g_losses["loss"],
            **{f"train/{k}": v for k, v in g_losses.items() if k != "loss"},
            **{f"train/{k}": v for k, v in d_losses.items()}
        }
        
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
        )

        # Log audio samples occasionally during training
        if self.global_step % 10000 == 0 and self.logger is not None:
            self.logger.experiment.log({
                "audio/train/src": wandb.Audio(
                    y[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
                "audio/train/tgt": wandb.Audio(
                    y_b1[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
                "audio/train/recon": wandb.Audio(
                    y_recon[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
                "audio/train/vc": wandb.Audio(
                    y_vc[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
            }, step=self.global_step)


        return {**d_losses, **g_losses}

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()


    def validation_step(self, batch: Any, batch_idx: int):
        (y_recon, y_vc, y, y_b1), (mel_y_recon_fine, mel_y_recon_coarse, mel_y), speaker_embs, (src_subj_id, tgt_subj_id) = self.model_step(batch)

        val_losses = {}
        val_losses['mel_fine_loss'] = self.charbonnier(mel_y_recon_fine, mel_y)
        val_losses['mel_coarse_loss'] = self.charbonnier(mel_y_recon_coarse, mel_y)
        # Speaker metrics
        val_losses.update(self._spk_consistency_losses(speaker_embs))
        val_losses.update(self._contrastive_loss(speaker_embs, src_subj_id, tgt_subj_id))

        # update and log metrics
        self.val_mel_loss(val_losses['mel_fine_loss'])
        self.log_dict(
            {
                "val/total/loss": val_losses["mel_fine_loss"],
                "val/mel/loss": val_losses["mel_fine_loss"],
                **{f"val/{k}": v for k, v in val_losses.items()},
            },
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        # Log audio samples from first validation batch
        if batch_idx == 0 and self.logger is not None:
            self.logger.experiment.log({
                "audio/val/src": wandb.Audio(
                    y[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
                "audio/val/tgt": wandb.Audio(
                    y_b1[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
                "audio/val/recon": wandb.Audio(
                    y_recon[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
                "audio/val/vc": wandb.Audio(
                    y_vc[0].detach().cpu().numpy().squeeze(),
                    sample_rate=self.hparams.sample_rate
                ),
            }, step=self.global_step)
        
        return {**val_losses}

    def on_validation_epoch_end(self):
        mel_loss = self.val_mel_loss.compute()  # get current val acc
        self.val_loss_best(mel_loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute())

    def test_step(self, batch: Any, batch_idx: int):
        (y_recon, y_vc, y, y_b1), (mel_y_recon_fine, mel_y_recon_coarse, mel_y), speaker_embs, (src_subj_id, tgt_subj_id) = self.model_step(batch)

        loss = self.charbonnier(mel_y_recon_fine, mel_y)
        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def on_test_epoch_end(self):
        pass


if __name__ == "__main__":
    m = VOSSASynthesizer()
    print(m)