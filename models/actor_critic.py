import argparse
from typing import Callable, Iterator, List, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from models.env import Env
import torch.nn.functional as F
from backbones.BERT.dataset_bert import BERTDataset

from tqdm import tqdm
from utils.data_utils import read_binary_file

from transformers import BertModel, BertTokenizer

SEP = "[SEP]"
USER = "[USER]"  # additional special token
BOT = "[BOT]"    # additional special token
ACTION = "[A]"
TOPIC = "[T]"
TARGET = "[TARGET]"
PATH = "[PATH]"

NEW_ADD_TOKENS = ["[USER]", "[BOT]","[A]","[T]","[TARGET]", "[PATH]"]

all_goals = read_binary_file("data/all_goals.pkl")
all_topics = read_binary_file("data/all_topics.pkl")

GOAL2ID = {k:id for id, k in enumerate(all_goals)}
TOPIC2ID = {k:id for id, k in enumerate(all_topics)}

ID2GOAL = {id:k for id, k in enumerate(all_goals)}
ID2TOPIC = {id:k for id, k in enumerate(all_topics)}


class Actor(nn.Module):
    """Policy network, for discrete action spaces, which returns a distribution and an action given an
    observation."""
    def __init__(self, actor_net):
        super(Actor, self).__init__()
        self.actor_net = actor_net

    def forward(self, states):
        logits = self.actor_net(states)
        distribution = Categorical(F.softmax(logits, dim=-1))
        return distribution, logits
    
    def get_log_prob(self, dis):
        action = dis.sample()
        log_prob =  dis.log_prob(action).unsqueeze(0)
        return action, log_prob
  

class Critic(nn.Module):
    def __init__(self, lm_size):
        super(Critic, self).__init__()
        
        self.lm_size = lm_size
        self.linear1 = nn.Linear(self.lm_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, states):
        output = F.relu(self.linear1(states))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


class ExperienceSourceDataset(IterableDataset):
    """Implementation from PyTorch Lightning Bolts: https://github.com/PyTorchLightning/lightning-
    bolts/blob/master/pl_bolts/datamodules/experience_source.py.
    Basic experience source dataset. Takes a generate_batch function that returns an iterator. The logic for the
    experience source and how the batch is generated is defined the Lightning model itself
    """

    def __init__(self, generate_batch: Callable):
        self.generate_batch = generate_batch

    def __iter__(self) -> Iterator:
        iterator = self.generate_batch()
        return iterator


class A2CPolicyNetwork(pl.LightningModule):
    """PyTorch Lightning implementation of PPO.
    Example:
        model = A2CPolicyNetwork("CartPole-v0")
    Train:
        trainer = Trainer()
        trainer.fit(model)
    """

    def __init__(
        self,
        train_dataset,
        args
    ) -> None:
        """
        Args:
            gamma: discount factor
            lam: advantage discount factor (lambda in the paper)
            lr_actor: learning rate of actor network
            lr_critic: learning rate of critic network
            max_episode_len: maximum number interactions (actions) in an episode
            batch_size:  batch_size when training network- can simulate number of policy updates performed per epoch
            steps_per_epoch: how many action-state pairs to rollout for trajectory collection per epoch
            nb_optim_iters: how many steps of gradient descent to perform on each batch
            clip_ratio: hyperparameter for clipping in the policy objective
        """
        super().__init__()

        if torch.cuda.is_available() and args.use_gpu:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Hyperparameters
        self.lr_actor = args.lr_actor
        self.lr_critic = args.lr_critic
        self.steps_per_epoch = args.steps_per_epoch
        self.nb_optim_iters = args.nb_optim_iters
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.lam = args.lam
        self.max_episode_len = args.max_episode_len
        self.clip_ratio = args.clip_ratio
        self.save_hyperparameters()

        # actor_backbone = BertModel.from_pretrained("bert-base-cased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")   
        special_tokens_dict = {'additional_special_tokens': NEW_ADD_TOKENS}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.bert_dataset = BERTDataset(None, self.tokenizer)

        self.env: Env = Env(train_dataset = train_dataset, tokenizer= self.tokenizer, args=args, device = device)
        ### value network
        ### actor network
        # self.actor = Actor(backbone = actor_backbone, n_goals = args.n_goals, n_topics = args.n_topics, lm_size=768, fc_size = 128)

        ### load the pretrained rl model
        self.temp = torch.load("logs/rl_pretrain/best_model.bin")
        ### freeze the model's parameters.
        self.temp.eval()
        for param in self.temp.backbone.parameters():
            param.requires_grad = False
        
        ### assign the same encoder for both actor and critic.
        self.encoder = self.temp.backbone

        ### assign the real actor as the output layer of the pretrained rl model
        self.actor = Actor(self.temp.layer)
        self.actor.train()
        self.actor.to(device)

        ### init the critic model
        self.critic = Critic(lm_size = 768)

        # self.batch_graphs = []
        self.batch_states = []
        self.batch_actions = []
        self.batch_adv = []
        self.batch_qvals = []
        self.batch_logp = []

        self.ep_rewards = []
        self.ep_values = []
        self.epoch_rewards = []

        self.episode_step = 0
        self.avg_ep_reward = 0
        self.avg_ep_len = 0
        self.avg_reward = 0

        self.state = None
        self.graph_embed = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Passes in a state x through the network and returns the policy and a sampled action.
        Args:
            x: environment state
            graph_embed: graph nodes represtation
        Returns:
            Tuple of policy and action
        """
        ### create the state for the actor and critic networks
        real_state = torch.LongTensor(x).to(self.device).unsqueeze(0)
        ### compute the hidden state
        real_state = self.encoder(real_state)[0]
        ### get the cls token embedding
        real_state = real_state[:, 0, :]
        ### compute the action and value
        distribution, logits = self.actor(real_state)
        value = self.critic(real_state)

        return distribution, logits, value

    def discount_rewards(self, rewards: List[float], discount: float) -> List[float]:
        """Calculate the discounted rewards of all rewards in list.
        Args:
            rewards: list of rewards/advantages
        Returns:
            list of discounted rewards/advantages
        """
        assert isinstance(rewards[0], float)

        cumul_reward = []
        sum_r = 0.0

        for r in reversed(rewards):
            sum_r = (sum_r * discount) + r
            cumul_reward.append(sum_r)

        return list(reversed(cumul_reward))

    def calc_advantage(self, rewards: List[float], values: List[float], last_value: float) -> List[float]:
        """Calculate the advantage given rewards, state values, and the last value of episode.
        Args:
            rewards: list of episode rewards
            values: list of state values from critic
            last_value: value of last state of episode
        Returns:
            list of advantages
        """
        rews = rewards + [last_value]
        vals = values + [last_value]
        # GAE
        delta = [rews[i] + self.gamma * vals[i + 1] - vals[i] for i in range(len(rews) - 1)]
        adv = self.discount_rewards(delta, self.gamma * self.lam)

        return adv

    def generate_trajectory_samples(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Contains the logic for generating trajectory data to train policy and value network
        Yield:
           Tuple of Lists containing tensors for states, actions, log probs, qvals and advantage
        """

        for step in tqdm(range(self.steps_per_epoch)):
            if self.state is None:
                self.state = self.env.reset()
            self.state = self.state
            # self.graph_embed = graph_embed.to(self.device)

            with torch.no_grad():
                distribution, logits, value = self(self.state)
                action, log_prob = self.actor.get_log_prob(distribution)

            next_state_dict, reward, done, _ = self.env.step(action)
            # graph_embed, pool_state = self.encode_state(next_state_dict)

            self.episode_step += 1

            # self.batch_graphs.append([1])
            self.batch_states.append(self.state)
            self.batch_actions.append(action)
            self.batch_logp.append(log_prob)

            self.ep_rewards.append(reward)
            self.ep_values.append(value.item())

            self.state = next_state_dict
            # self.graph_embed = graph_embed

            epoch_end = step == (self.steps_per_epoch - 1)
            terminal = len(self.ep_rewards) == self.max_episode_len

            if epoch_end or done or terminal:
                # if trajectory ends abtruptly, boostrap value of next state
                if (terminal or epoch_end) and not done:
                    with torch.no_grad():
                        _, _, value = self(self.state)
                        last_value = value.item()
                        steps_before_cutoff = self.episode_step
                else:
                    last_value = 0
                    steps_before_cutoff = 0

                # discounted cumulative reward
                self.batch_qvals += self.discount_rewards(self.ep_rewards + [last_value], self.gamma)[:-1]
                # advantage
                self.batch_adv += self.calc_advantage(self.ep_rewards, self.ep_values, last_value)
                # logs
                self.epoch_rewards.append(sum(self.ep_rewards))
                # reset params
                self.ep_rewards = []
                self.ep_values = []
                self.episode_step = 0
                self.state = None

            if epoch_end:
                train_data = zip(
                    self.batch_states, self.batch_actions, self.batch_logp, self.batch_qvals, self.batch_adv
                )

                for state, action, logp_old, qval, adv in train_data:
                    yield state, action, logp_old, qval, adv

                # self.batch_graphs.clear()
                self.batch_states.clear()
                self.batch_actions.clear()
                self.batch_adv.clear()
                self.batch_logp.clear()
                self.batch_qvals.clear()

                # logging
                self.avg_reward = sum(self.epoch_rewards) / self.steps_per_epoch

                # if epoch ended abruptly, exlude last cut-short episode to prevent stats skewness
                epoch_rewards = self.epoch_rewards
                if not done:
                    epoch_rewards = epoch_rewards[:-1]

                total_epoch_reward = sum(epoch_rewards)
                nb_episodes = len(epoch_rewards)

                self.avg_ep_reward = total_epoch_reward / nb_episodes
                self.avg_ep_len = (self.steps_per_epoch - steps_before_cutoff) / nb_episodes


                self.epoch_rewards.clear()

    def actor_loss(self, state_list, action_list, logp_old, qval, adv) -> torch.Tensor:
        
        ### construct the states and input to the actor model
        # real_states = [ torch.LongTensor(x).to(self.device).unsqueeze(0) for x in state_list ]
        real_states = self.bert_dataset.rl_collate(state_list).to(self.device)
        real_states = self.encoder(real_states)[0]
        real_states = real_states[:, 0, :]
        dis, _ = self.actor(real_states)
        a = dis.sample()
        logp = dis.log_prob(a)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_actor = -(torch.min(ratio * adv, clip_adv)).mean()
        return loss_actor

    def critic_loss(self, state_list, action, logp_old, qval, adv) -> torch.Tensor:
        real_states = self.bert_dataset.rl_collate(state_list).to(self.device)
        real_states = self.encoder(real_states)[0]
        real_states = real_states[:, 0, :]
        value = self.critic(real_states)
        loss_critic = (qval - value).pow(2).mean()
        return loss_critic

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx, optimizer_idx):
        """Carries out a single update to actor and critic network from a batch of replay buffer.
        Args:
            batch: batch of replay buffer/trajectory data
            batch_idx: not used
            optimizer_idx: idx that controls optimizing actor or critic network
        Returns:
            loss
        """
        state, action, old_logp, qval, adv = batch

        # normalize advantages
        adv = (adv - adv.mean()) / adv.std()

        self.log("avg_ep_len", float(self.avg_ep_len), prog_bar=True, on_step=False, batch_size=self.batch_size, on_epoch=True)
        self.log("avg_ep_reward", float(self.avg_ep_reward), prog_bar=True, on_step=False, batch_size=self.batch_size, on_epoch=True)
        self.log("avg_reward", float(self.avg_reward), prog_bar=True, on_step=False, batch_size=self.batch_size, on_epoch=True)

        if optimizer_idx == 0:
            loss_actor = self.actor_loss(state, action, old_logp, qval, adv)
            self.log("loss_actor", loss_actor, on_step=False, batch_size=self.batch_size, on_epoch=True, prog_bar=True, logger=True)
            return loss_actor

        if optimizer_idx == 1:
            loss_critic = self.critic_loss(state, action, old_logp, qval, adv)
            self.log("loss_critic", loss_critic, on_step=False, batch_size=self.batch_size, on_epoch=True, prog_bar=False, logger=True)
            return loss_critic

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        return optimizer_actor, optimizer_critic

    def optimizer_step(self, *args, **kwargs):
        """Run 'nb_optim_iters' number of iterations of gradient descent on actor and critic for each data
        sample."""
        for _ in range(self.nb_optim_iters):
            super().optimizer_step(*args, **kwargs)

    def collate(self, batch):
        state = []
        action = []  # torch.Size([2, 1112])
        old_logp = [] # 2 1
        qval = [] # 2
        adv = [] # 2
        for row in batch:
            state.append(row[0])
            action.append(row[1])
            old_logp.append(row[2])
            qval.append(row[-2])
            adv.append(row[-1])
        return state, torch.stack(action), torch.stack(old_logp), torch.tensor(qval), torch.tensor(adv)

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = ExperienceSourceDataset(self.generate_trajectory_samples)
        # dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, collate_fn=self.collate)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self._dataloader()


if __name__ == "__main__":
    from pytorch_lightning.callbacks import ModelCheckpoint, progress

    pl.seed_everything(0)
    tb_logger = pl.loggers.TensorBoardLogger('logs_rl/', name='')
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        save_last=True,
        verbose=True,
        filename='best',
        monitor='avg_ep_reward',
        mode='max'
    )
    bar_callback = progress.TQDMProgressBar(refresh_rate=50)
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parser = A2CPolicyNetwork.add_model_specific_args(parent_parser)
    args = parser.parse_args()
    model = A2CPolicyNetwork(**vars(args))

    args.gpus = [0]
    args.logger = tb_logger
    args.detect_anomaly = True
    args.gradient_clip_val = 0.5
    args.callbacks = [checkpoint_callback, bar_callback]
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model)
