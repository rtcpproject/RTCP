import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
from tqdm import tqdm
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup


class Trainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,
            model, 
            train_loader, 
            dev_loader, 
            log_dir, 
            log_steps, 
            validate_steps, 
            num_epochs, 
            lr, 
            warm_up_ratio=0.1, 
            weight_decay=0.01, 
            max_grad_norm=0.5
        ):

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_ratio = warm_up_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warm_up_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        """
        Train the model.
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Evaluate every {} batches.".format(self.validate_steps))

        best_model_store_path = os.path.join(self.log_dir, "best_model.bin")
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(tqdm(self.train_loader)):
                self.model.train()
                model_output = self.model(inputs)
                loss = model_output["loss"]
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if batch_step > 0 and batch_step % self.log_steps == 0:
                    logging.info("Batch Step: {}\tloss: {:.3f}".format(batch_step, loss.item()))

                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    logging.info("Evaluation Acc: {:.3f}, Evaluation Topic Acc: {:.3f} , loss: {:.3f}".format(
                        predicts_dict["avg_acc"], predicts_dict["avg_topic_acc"], predicts_dict["avg_loss"])
                    )

                    if predicts_dict["avg_topic_acc"] > self.best_metric:
                        self.best_metric = predicts_dict["avg_topic_acc"]
                        logging.info("Epoch {} Batch Step {} -- Best Topic Acc: {:.3f} -- loss: {:.3f}".format(epoch + 1, batch_step, self.best_metric, predicts_dict['avg_loss']))
                        torch.save(self.model, best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
            predicts_dict = self.evaluate(loader=self.dev_loader)
            if predicts_dict["avg_topic_acc"] > self.best_metric:
                self.best_metric = predicts_dict["avg_topic_acc"]
                logging.info("Epoch {} Best Avg Topic Acc: {:.3f} -- loss: {:.3f}".format(epoch, self.best_metric, predicts_dict['avg_loss']))
                torch.save(self.model, best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(self.log_dir, "model_epoch_%d.bin"%(epoch + 1))
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def evaluate(self, loader):
        """
        Use trained model to perform evaluation.
        Args:
            loader: the DataLoader containing the data to run evaluation.         
        Returns:
            avg_acc: average accuracy.
            avg_loss: average loss.
        """
#         self.model.eval()
        
        total_acc = 0.0
        total_topic_acc = 0.0
        count_tok = 0.0
        loss = []
        for inputs in tqdm(loader):
            with torch.no_grad():
                output = self.model(inputs)
                
                acc = output["acc"]
                total_tokens = output["total_tokens"]
                total_acc += acc
                count_tok += total_tokens
                
                topic_acc = output['topic_acc']
                total_topic_acc += topic_acc
                
                loss.append(float(output["loss"]))
        avg_acc = total_acc / count_tok
        avg_topic_acc = total_topic_acc/ count_tok
        avg_loss = np.mean(loss)
        
        return_dict = {
            "avg_acc": avg_acc,
            "avg_topic_acc": avg_topic_acc,
            "avg_loss": avg_loss,
        }
        return return_dict


class RLTrainer(object):
    """
    Trainer with `train` and `evaluate` functions.
    """
    def __init__(self,
            model, 
            train_loader, 
            dev_loader, 
            log_dir, 
            log_steps, 
            validate_steps, 
            num_epochs, 
            lr, 
            warm_up_ratio=0.1, 
            weight_decay=0.01, 
            max_grad_norm=0.5
        ):

        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.log_dir = log_dir
        self.log_steps = log_steps
        self.validate_steps = validate_steps
        self.num_epochs = num_epochs
        self.lr = lr
        self.warm_up_ratio = warm_up_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.criterion = nn.CrossEntropyLoss()

        total_steps = len(train_loader) * self.num_epochs
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
            num_warmup_steps=self.warm_up_ratio * total_steps, 
            num_training_steps=total_steps)
        self.best_metric = 0.0
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def train(self):
        """
        Train the model.
        """
        logging.info("Total batches per epoch : {}".format(len(self.train_loader)))
        logging.info("Evaluate every {} batches.".format(self.validate_steps))

        best_model_store_path = os.path.join(self.log_dir, "best_model.bin")
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}:".format(epoch + 1))
            for batch_step, inputs in enumerate(tqdm(self.train_loader)):
                self.model.train()
                _, logits = self.model(inputs["path"])
                loss = self.criterion(logits, inputs["label"])
                loss.backward()
                if self.max_grad_norm > 0:
                    nn_utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if batch_step > 0 and batch_step % self.log_steps == 0:
                    logging.info("Batch Step: {}\tloss: {:.3f}".format(batch_step, loss.item()))

                if batch_step > 0 and batch_step % self.validate_steps == 0:
                    logging.info("Evaluating...")
                    predicts_dict = self.evaluate(loader=self.dev_loader)
                    logging.info("Evaluation Acc: {:.3f}, Evaluation Topic Acc: {:.3f} , loss: {:.3f}".format(
                        predicts_dict["avg_acc"], predicts_dict["avg_topic_acc"], predicts_dict["avg_loss"])
                    )
                    # ##wandb logging
                    # wandb.log({'train/TCP_loss': loss.item()})
                    # wandb.log({'val/TCP_loss': predicts_dict["avg_loss"]})
                    # wandb.log({'val/TCP_accuracy': predicts_dict["avg_acc"]})
                    # wandb.log({'val/TCP_topic_accuracy': predicts_dict["avg_topic_acc"]})

                    if predicts_dict["avg_topic_acc"] > self.best_metric:
                        self.best_metric = predicts_dict["avg_topic_acc"]
                        logging.info("Epoch {} Batch Step {} -- Best Topic Acc: {:.3f} -- loss: {:.3f}".format(epoch + 1, batch_step, self.best_metric, predicts_dict['avg_loss']))
                        torch.save(self.model, best_model_store_path)
                        logging.info("Saved to [%s]" % best_model_store_path)
            predicts_dict = self.evaluate(loader=self.dev_loader)
            if predicts_dict["avg_topic_acc"] > self.best_metric:
                self.best_metric = predicts_dict["avg_topic_acc"]
                logging.info("Epoch {} Best Avg Topic Acc: {:.3f} -- loss: {:.3f}".format(epoch, self.best_metric, predicts_dict['avg_loss']))
                torch.save(self.model, best_model_store_path)
                logging.info("Saved to [%s]" % best_model_store_path)
            logging.info("Epoch {} training done.".format(epoch + 1))
            model_to_save = os.path.join(self.log_dir, "model_epoch_%d.bin"%(epoch + 1))
            torch.save(self.model, model_to_save)
            logging.info("Saved to [%s]" % model_to_save)

    def evaluate(self, loader):
        """
        Use trained model to perform evaluation.
        Args:
            loader: the DataLoader containing the data to run evaluation.         
        Returns:
            avg_acc: average accuracy.
            avg_loss: average loss.
        """
#         self.model.eval()
        
        total_acc = 0.0
        total_topic_acc = 0.0
        count_tok = 0.0
        loss = []
        for inputs in tqdm(loader):
            with torch.no_grad():
                _, output = self.model(inputs["path"])
                labels = inputs["label"]
                pred = output.argmax(-1)
                acc = (torch.eq(pred,labels).float()).sum().item()
                total_acc += acc
                count_tok += pred.shape[0]
        avg_acc = total_acc / count_tok
        avg_topic_acc = avg_acc
        avg_loss = np.mean(loss)
        
        return_dict = {
            "avg_acc": avg_acc,
            "avg_topic_acc": avg_topic_acc,
            "avg_loss": 0,
        }
        return return_dict