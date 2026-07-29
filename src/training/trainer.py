"""
===============================================================
Trainer
Centralized Training Engine
===============================================================
"""

import copy
import torch
from tqdm import tqdm

from src.training.metrics import calculate_metrics


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        scheduler,
        device
    ):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

    #########################################################

    def train_one_epoch(self, dataloader):

        self.model.train()

        running_loss = 0

        for images, labels in tqdm(
            dataloader,
            leave=False
        ):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    #########################################################

    @torch.no_grad()

    def validate(self, dataloader):

        self.model.eval()

        running_loss = 0

        y_true = []
        y_pred = []
        y_prob = []

        for images, labels in tqdm(
            dataloader,
            leave=False
        ):

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels
            )

            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]

            preds = outputs.argmax(1)

            running_loss += loss.item()

            y_true.extend(labels.cpu().numpy())

            y_pred.extend(preds.cpu().numpy())

            y_prob.extend(probs.cpu().numpy())

        metrics = calculate_metrics(
            y_true,
            y_pred,
            y_prob
        )

        val_loss = running_loss / len(dataloader)

        return val_loss, metrics

    #########################################################

    def fit(

        self,

        train_loader,

        val_loader,

        epochs,

        checkpoint_path

    ):

        best_auc = 0

        history = []

        best_weights = copy.deepcopy(
            self.model.state_dict()
        )

        for epoch in range(epochs):

            print()

            print("=" * 60)

            print(f"Epoch {epoch+1}/{epochs}")

            train_loss = self.train_one_epoch(
                train_loader
            )

            val_loss, metrics = self.validate(
                val_loader
            )

            self.scheduler.step(val_loss)

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Val Loss   : {val_loss:.4f}")

            print(f"Accuracy   : {metrics['accuracy']:.4f}")

            print(f"AUC        : {metrics['auc']:.4f}")

            history.append({

                "epoch": epoch + 1,

                "train_loss": train_loss,

                "val_loss": val_loss,

                **metrics

            })

            if metrics["auc"] > best_auc:

                best_auc = metrics["auc"]

                best_weights = copy.deepcopy(

                    self.model.state_dict()

                )

                torch.save(

                    best_weights,

                    checkpoint_path

                )

                print()

                print("Best Model Saved")

        self.model.load_state_dict(

            best_weights

        )

        return history