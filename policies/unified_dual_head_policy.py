"""
共享底层 CNN + 双动作头策略：
- Mario 头
- CoinRun 头

通过观测固定像素位的 game_id 路由动作头：
- obs[..., 0, 0, 0] 接近 0   -> Mario
- obs[..., 0, 0, 0] 接近 255 -> CoinRun
"""
import torch as th
import torch.nn as nn

from stable_baselines3.common.policies import ActorCriticCnnPolicy


class UnifiedDualHeadPolicy(ActorCriticCnnPolicy):
    """
    完全共享底层特征提取器，仅在最终动作层拆分为两头。
    """

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        action_dim = int(self.action_space.n)
        self.mario_action_net = nn.Linear(latent_dim_pi, action_dim)
        self.coinrun_action_net = nn.Linear(latent_dim_pi, action_dim)
        if self.ortho_init:
            nn.init.orthogonal_(self.mario_action_net.weight, gain=0.01)
            nn.init.zeros_(self.mario_action_net.bias)
            nn.init.orthogonal_(self.coinrun_action_net.weight, gain=0.01)
            nn.init.zeros_(self.coinrun_action_net.bias)

    @staticmethod
    def _coinrun_mask_from_obs(obs: th.Tensor) -> th.Tensor:
        """
        从观测中提取 coinrun 掩码（True=coinrun，False=mario）。
        支持 CHW 输入，mask 形状为 (batch,)。
        """
        if obs.dim() != 4:
            return th.zeros((obs.shape[0],), dtype=th.bool, device=obs.device)
        marker = obs[:, 0, 0, 0]
        if th.is_floating_point(marker):
            max_val = float(marker.max().detach().item())
            threshold = 0.5 if max_val <= 1.0 + 1e-6 else 127.5
            return marker > threshold
        return marker > 127

    def _get_latent(self, obs: th.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def _get_action_dist(self, latent_pi: th.Tensor, obs: th.Tensor):
        mario_logits = self.mario_action_net(latent_pi)
        coinrun_logits = self.coinrun_action_net(latent_pi)
        coinrun_mask = self._coinrun_mask_from_obs(obs).unsqueeze(1)
        action_logits = th.where(coinrun_mask, coinrun_logits, mario_logits)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        latent_pi, latent_vf = self._get_latent(obs)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist(latent_pi, obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def get_distribution(self, obs: th.Tensor):
        latent_pi, _ = self._get_latent(obs)
        return self._get_action_dist(latent_pi, obs)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        latent_pi, latent_vf = self._get_latent(obs)
        distribution = self._get_action_dist(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy
