from copy import copy

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
            "mel_loss_MPD": mel_loss_MPD,
            "mel_loss_MSD": mel_loss_MSD,
            "gan_loss_MPD": gan_loss_MPD,
            "gan_loss_MSD": gan_loss_MSD,
            "loss": loss_discriminator_MPD
            + loss_discriminator_MSD
            + loss_generator_MPD
            + loss_generator_MSD,
        }

        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
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
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
