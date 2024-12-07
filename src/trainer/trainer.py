from copy import copy

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)

        batch_transformed = self.transform_batch(
            copy(batch)
        )  # transform batch on device -- faster
        batch["input_spec"] = batch_transformed["input"]

        # metric_funcs = self.metrics["inference"]
        if self.is_train:
            # metric_funcs = self.metrics["train"]
            self.optimizer_generator.zero_grad()
            self.optimizer_discriminator.zero_grad()

        outputs_generator = self.generator(**batch)
        batch.update(outputs_generator)

        (
            hidden_features1_MPD,
            real_outputs_MPD,
            hidden_features2_MPD,
            generated_outputs_MPD,
        ) = self.discriminatorMPD(
            real_wav=batch["input"],
            generated_wav=batch["generated_wav"].clone().detach(),
        )
        (
            hidden_features1_MSD,
            real_outputs_MSD,
            hidden_features2_MSD,
            generated_outputs_MSD,
        ) = self.discriminatorMSD(
            real_wav=batch["input"],
            generated_wav=batch["generated_wav"].clone().detach(),
        )

        loss_discriminator_MPD = self.loss_discriminator(
            real=real_outputs_MPD, fake=generated_outputs_MPD
        )
        loss_discriminator_MSD = self.loss_discriminator(
            real=real_outputs_MSD, fake=generated_outputs_MSD
        )

        (
            hidden_features1_MPD,
            real_outputs_MPD,
            hidden_features2_MPD,
            generated_outputs_MPD,
        ) = self.discriminatorMPD(
            real_wav=batch["input"], generated_wav=batch["generated_wav"]
        )
        (
            hidden_features1_MSD,
            real_outputs_MSD,
            hidden_features2_MSD,
            generated_outputs_MSD,
        ) = self.discriminatorMSD(
            real_wav=batch["input"], generated_wav=batch["generated_wav"]
        )

        (
            loss_generator_MPD,
            mathcing_loss_MPD,
            mel_loss_MPD,
            gan_loss_MPD,
        ) = self.loss_generator(
            hidden_features1=hidden_features1_MPD,
            hidden_features2=hidden_features2_MPD,
            wav1=batch["input"],
            wav2=batch["generated_wav"],
            fake=generated_outputs_MPD,
        )

        (
            loss_generator_MSD,
            mathcing_loss_MSD,
            mel_loss_MSD,
            gan_loss_MSD,
        ) = self.loss_generator(
            hidden_features1=hidden_features1_MSD,
            hidden_features2=hidden_features2_MSD,
            wav1=batch["input"],
            wav2=batch["generated_wav"],
            fake=generated_outputs_MSD,
        )

        all_losses = {
            "loss_discriminator_MPD": loss_discriminator_MPD,
            "loss_discriminator_MSD": loss_discriminator_MSD,
            "loss_generator_MPD": loss_generator_MPD,
            "loss_generator_MSD": loss_generator_MSD,
            "matching_loss_MPD": mathcing_loss_MPD,
            "matching_loss_MSD": mathcing_loss_MSD,
            "mel_loss": mel_loss_MPD + mel_loss_MSD,
            "gan_loss_MPD": gan_loss_MPD,
            "gan_loss_MSD": gan_loss_MSD,
            "loss_disriminator": loss_discriminator_MPD + loss_discriminator_MSD,
            "loss_generator": loss_generator_MPD + loss_generator_MSD,
        }

        batch.update(all_losses)

        if self.is_train:
            batch["loss_disriminator"].backward()
            batch["loss_generator"].backward()
            self._clip_grad_norm()
            self.optimizer_generator.step()
            self.optimizer_discriminator.step()
            if self.lr_scheduler_generator is not None:
                self.lr_scheduler_generator.step()
            if self.lr_scheduler_discriminator is not None:
                self.lr_scheduler_discriminator.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        # for met in metric_funcs:
        #     metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":
            self.log_spectrogram(
                batch["input_spec"][0].detach().cpu(), "original spec train"
            )
            generated_spec = self.spec(batch["generated_wav"][0].detach()).cpu()
            self.log_spectrogram(generated_spec, "generated spec train")
            # self.log_spectrogram(batch['input_spec'], "input_spec")
            self.log_audio(
                batch["input"][0].detach().cpu(),
                batch["generated_wav"][0].detach().cpu(),
                "train",
            )
        else:
            self.log_spectrogram(
                batch["input_spec"][0].detach().cpu(), "original spec val"
            )
            generated_spec = self.spec(batch["generated_wav"][0].detach()).cpu()
            self.log_spectrogram(generated_spec, "generated spec val")
            self.log_audio(
                batch["input"][0].detach().cpu(),
                batch["generated_wav"][0].detach().cpu(),
                "val",
            )

    def log_spectrogram(self, spectrogram, name):
        image = plot_spectrogram(spectrogram)
        self.writer.add_image(name, image)

    def log_audio(self, wav_true, wav_generated, part):
        real_wav = self.writer.wandb.Audio(
            wav_true.squeeze(0).detach().cpu().numpy(), sample_rate=22050
        )
        generated_wav = self.writer.wandb.Audio(
            wav_generated.squeeze(0).detach().cpu().numpy(), sample_rate=22050
        )

        self.writer.add_table(
            "audio",
            pd.DataFrame.from_dict(
                {
                    1: {
                        f"real_wav_{part}": real_wav,
                        f"generated_wav_{part}": generated_wav,
                    }
                },
                orient="index",
            ),
        )
