from typing import List, Tuple, Dict

import gym
import numpy as np

import agent
import sim


class KosciEnv(gym.Env):
    """
    ## Actions ##

    There are n decisions to take each turn, n being the number of dice. 1 means the agent decided to reroll that die,
    0 means that the die was left as it is. All 0s effectively means that the agent decided to accept the roll and take
    the intermediate score. Note that most of the time some actions are prohibited by the rules of the game. The action
    space does not change so it is possible to take an illegal action, but that is severely penalized.

    ## Rewards ##

    We reward the agent with points related to what score was added at the end of their turn.
    We don't care about intermediate scores. This rewards the most actually taking what you rolled, which is
    the essential part. The downside is that there's no immediate feedback after each roll so it may be harder to learn.

    If a foul roll occurred (no points to take after the first roll), we gently move forward and don't give any penalty
    for bad luck. Besides, that would technically penalize sticking to the rules and accepting fate.

    For breaking the rules we penalize the agent and immediately end the game.

    We give rewards (or penalties!) for "overtaking" other players, i.e. scoring enough points in a turn to go from
    a score behind theirs to ahead of theirs. If the resulting distance between scores is big enough (> set boundary),
    a scaled reward is given. If it's less than the the boundary, a scaled penalty is given. The bigger the distance is,
    the better the reward is (and vice versa for penalties).

    For finishing the game we reward the agent with a reward that linearly diminishes with each turn.
    """
    metadata = {'render.modes': ['human']}

    ILLEGAL_ACTION_PENALTY = -10000

    GAME_FINISHED_REWARD_START = 1000
    GAME_FINISHED_REWARD_DECREASE_RATE = 10

    OVERTAKE_REWARD_BASE = 20
    SUSCEPTIBLE_TO_OVERTAKE_PENALTY_BASE = -15
    OVERTAKE_SAFETY_DISTANCE = 50

    CONTROLLED_PLAYER_IDX = 0

    def _create_opponents_agents(self) -> List:
        opponents_agents = [agent.AgentFactory.make_agent('niegardzący') for _ in range(self.n_players)]
        opponents_agents[self.CONTROLLED_PLAYER_IDX] = agent.AgentFactory.make_agent('dummy')
        return opponents_agents

    def __init__(self, n_players: int):
        super().__init__()

        self.n_players = n_players
        self.opponents_agents = self._create_opponents_agents()
        self.game = None
        self.action_space = gym.spaces.Box(0, 1, [sim.Game.N_DICE], np.int8)
        self.observation_space = gym.spaces.Dict({
            'dice': gym.spaces.Box(1, 6, [sim.Game.N_DICE], np.int8),
            'n_locked_dice': gym.spaces.Discrete(sim.Game.N_DICE + 1),
            'score_in_memory': gym.spaces.Box(0, 1000, [1], np.int16),
            'player_score': gym.spaces.Box(-1000, 1000, [1], np.int16),
            'opponents_score': gym.spaces.Box(-1000, 1000, [n_players - 1], np.int16),
            'player_entered': gym.spaces.Box(0, 1, [1], np.int8),
            'opponents_entered': gym.spaces.Box(0, 1, [n_players - 1], np.int8)
        })

    def _accept_roll(self) -> Tuple[float, bool]:
        score_before = self._get_current_player_score()
        self.game.accept_roll()
        score = self._get_current_player_score()
        reward = (score - score_before)
        if reward > 0:
            reward += self._get_overtakes_rewards_penalties()
        done = self._handle_turn_end()
        return reward, done

    def _do_full_reroll(self):
        self.game.do_full_reroll()

    def _do_partial_reroll(self, action):
        cut_bool_indices = action[len(self.game.current_player_kept_dice):]
        int_indices = [i for i, b in enumerate(cut_bool_indices) if b]
        self.game.do_partial_reroll(int_indices)

    def _act(self, action) -> Tuple[float, bool]:
        action = np.round(action)
        reward = 0
        done = False
        if np.all(action == 0):
            reward, done = self._accept_roll()
        else:
            if np.all(action == 1):
                self._do_full_reroll()
            else:
                self._do_partial_reroll(action)

            if self.game.turn_finished:
                done = self._handle_turn_end()

        if done:
            reward += self._get_game_over_reward()
        return reward, done

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        try:
            reward, done = self._act(action)
        except sim.GameException as ex:
            print(f'Rule violation occurred: {ex}')
            reward = self.ILLEGAL_ACTION_PENALTY
            done = True
        observation = self._extract_observation()
        info = {}
        return observation, reward, done, info

    def reset(self) -> Dict:
        self.game = sim.Game(self.n_players)
        self._let_opponents_before_act()
        self.game.start_next_turn()
        observation = self._extract_observation()
        return observation

    def _get_current_player_score(self) -> int:
        return self.game.players_scores[self.game.current_player_idx]

    def _get_other_players_scores(self) -> np.ndarray:
        return np.delete(self.game.players_scores, self.game.current_player_idx)

    def _get_current_player_entered(self) -> bool:
        return self.game.players_entered[self.game.current_player_idx]

    def _get_other_players_entered(self) -> np.ndarray:
        return np.delete(self.game.players_entered, self.game.current_player_idx)

    def _extract_observation(self) -> Dict:
        assert self.game.current_player_idx == self.CONTROLLED_PLAYER_IDX
        return {
            'dice': np.append(self.game.current_player_kept_dice, self.game.current_player_new_dice).astype(np.int8),
            'n_locked_dice': len(self.game.current_player_kept_dice),
            'score_in_memory': np.array([self.game.current_player_score_in_memory], np.int16),
            'player_score': np.array([self._get_current_player_score()], np.int16),
            'opponents_score': self._get_other_players_scores().astype(np.int16),
            'player_entered': np.array([self._get_current_player_entered()], np.int8),
            'opponents_entered': self._get_other_players_entered().astype(np.int8)
        }

    def _let_opponent_act(self, idx):
        self.game.start_next_turn()
        agent = self.opponents_agents[idx]
        try:
            agent.act(self.game)
        except sim.GameException as ex:
            print(f'Opponent agent [{idx}] {str(agent)} tried to perform an illegal action: {ex}')
        try:
            assert self.game.turn_finished
        except AssertionError:
            print(f'Opponent agent [{idx}] {str(agent)} did not finish their turn properly')

    def _let_opponents_before_act(self):
        for idx in range(self.CONTROLLED_PLAYER_IDX):
            self._let_opponent_act(idx)

    def _let_opponents_after_act(self):
        for idx in range(self.CONTROLLED_PLAYER_IDX + 1, self.game.n_players):
            self._let_opponent_act(idx)

    def _handle_turn_end(self) -> bool:
        if self.game.is_over():
            return True
        else:
            self._let_opponents_after_act()
            self._let_opponents_before_act()
            self.game.start_next_turn()
            return False

    def _get_game_over_reward(self) -> int:
        return self.GAME_FINISHED_REWARD_START - (self.game.n_rounds - 1) * self.GAME_FINISHED_REWARD_DECREASE_RATE

    def _get_overtake_penalty_reward_boundary(self) -> float:
        return self.OVERTAKE_SAFETY_DISTANCE / 2  # boundary between giving rewards and penalties set halfway

    def _get_capped_overtake_distance(self, player_score: int, opponent_score: int) -> int:
        assert player_score >= opponent_score
        distance = opponent_score - player_score
        return max(distance, self.OVERTAKE_SAFETY_DISTANCE)

    def _get_overtake_reward_or_penalty(self, player_score: int, opponent_score: int) -> float:
        capped_distance = self._get_capped_overtake_distance(player_score, opponent_score)
        penalty_reward_boundary = self._get_overtake_penalty_reward_boundary()
        if capped_distance < penalty_reward_boundary:
            penalty_factor = (penalty_reward_boundary - capped_distance) / penalty_reward_boundary
            return -self.SUSCEPTIBLE_TO_OVERTAKE_PENALTY_BASE * penalty_factor
        else:
            reward_factor = (capped_distance - penalty_reward_boundary) / (self.OVERTAKE_SAFETY_DISTANCE - penalty_reward_boundary)
            return self.OVERTAKE_REWARD_BASE * reward_factor

    def _get_overtakes_rewards_penalties(self):
        delta = 0
        for idx, occurred in enumerate(self.game.overtakes):
            if not occurred:
                continue
            delta += self._get_overtake_reward_or_penalty(self.game.players_scores[self.CONTROLLED_PLAYER_IDX],
                                                          self.game.players_scores[idx])
        return delta

    def render(self, mode='human'):
        pass
