import logging
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

    ILLEGAL_ACTION_REACTION = 'penalize'  # retry, penalize
    ILLEGAL_ACTION_PENALTY = -1000

    GAME_WON_REWARD_BASE = 3000
    GAME_WON_REWARD_DECREASE_RATE = 10
    GAME_WON_REWARD_PERSISTING_BONUS = 100

    OVERTAKE_REWARD_BASE = 20
    SUSCEPTIBLE_TO_OVERTAKE_PENALTY_BASE = -15
    OVERTAKE_SAFETY_DISTANCE = 50

    BAD_GAME_POINTS_THRESHOLD = -1000
    BAD_GAME_PENALTY = -1000

    OPPONENT_GAME_OVER_PENALTY = -2000

    CONTROLLED_PLAYER_IDX = 0

    # every player and opponent score must not go beyond those limits
    MIN_SCORE = -2000
    MAX_SCORE = 2000

    def _create_opponents_agents(self) -> List:
        opponents_agents_types = ['inactive', 'dummy', 'niegardzący', 'moderately_lucky', 'feeling_lucky']
        assert len(opponents_agents_types) == self.n_players
        assert opponents_agents_types[self.CONTROLLED_PLAYER_IDX] == 'inactive'
        opponents_agents = [agent.AgentFactory.make_agent(t) for t in opponents_agents_types]
        opponents_agents[self.CONTROLLED_PLAYER_IDX] = agent.AgentFactory.make_agent('inactive')
        return opponents_agents

    @staticmethod
    def _create_action_space() -> gym.spaces.MultiBinary:
        # 1 means the player wants to reroll a dice on that position
        return gym.spaces.MultiBinary(sim.Game.N_DICE)

    @staticmethod
    def _create_observation_space() -> gym.spaces.Dict:
        def create_score():
            return gym.spaces.Discrete(KosciEnv.MAX_SCORE - KosciEnv.MIN_SCORE + 1)

        return gym.spaces.Dict({
            # player's current dice scores
            'dice': gym.spaces.MultiDiscrete([6] * sim.Game.N_DICE, np.int8),
            # player's number of locked dice (they are always placed at the beginning of dice list)
            'n_locked_dice': gym.spaces.Discrete(sim.Game.N_DICE + 1),
            # player's score in memory
            'score_in_memory': create_score(),
            # player / RL agent score
            'player_score': create_score(),
            # score of the next best OR the next worse opponent (if the player is #1)
            'next_opponent_score': create_score(),
            # score of the best opponent
            'best_opponent_score': create_score(),
            # score of the worst opponent
            'worst_opponent_score': create_score(),
            # whether the player entered the game
            'player_entered': gym.spaces.MultiBinary(1),
            # whether any opponent entered the game
            'any_opponent_entered': gym.spaces.MultiBinary(1)
        })

    def __init__(self, n_players: int):
        assert(n_players > 0)

        super().__init__()

        self.n_players = n_players
        self.opponents_agents = self._create_opponents_agents()
        self.game = None
        self.action_space = self._create_action_space()
        self.observation_space = self._create_observation_space()

    def _accept_roll(self) -> Tuple[float, bool]:
        score_before = self._get_current_player_score()
        self.game.accept_roll()
        score = self._get_current_player_score()
        if score < self.BAD_GAME_POINTS_THRESHOLD:
            reward = self.BAD_GAME_PENALTY
            done = True
            return reward, done
        reward = (score - score_before)
        if reward > 0:
            reward += self._get_overtakes_rewards_penalties()
        done = self._handle_turn_end()
        return reward, done

    def _do_full_reroll(self):
        self.game.do_full_reroll()

    def _do_partial_reroll(self, action):
        kept_dice_num = len(self.game.current_player_kept_dice)
        if np.any(action[:kept_dice_num]):
            raise sim.GameException(sim.ErrorType.BAD_REROLL_CHOICE, 'One of kept dice chosen for a reroll')

        cut_bool_indices = action[kept_dice_num:]
        int_indices = np.array([i for i, b in enumerate(cut_bool_indices) if b])
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
            reward += self._get_game_over_reward_or_penalty()
        return reward, done

    def _deal_with_rule_violation(self, violation):
        logging.debug(f'Rule violation occurred: {violation}')
        if self.ILLEGAL_ACTION_REACTION == 'retry':
            return 0, False
        if self.ILLEGAL_ACTION_REACTION == 'penalize':
            return self.ILLEGAL_ACTION_PENALTY, True
        raise ValueError(f'Unknown illegal action reaction type: {self.ILLEGAL_ACTION_REACTION}')

    def step(self, action) -> Tuple[Dict, float, bool, Dict]:
        try:
            reward, done = self._act(action)
        except sim.GameException as ex:
            reward, done = self._deal_with_rule_violation(ex)
        observation = self._extract_observation() if not done else self.observation_space.sample()  # TODO this else could be improved
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

        player_score = int(self._get_current_player_score())
        opponents_scores = self._get_other_players_scores()
        opponents_sorted_i = np.argsort(opponents_scores)
        best_opponent_score = int(opponents_scores[opponents_sorted_i[-1]])
        worst_opponent_score = int(opponents_scores[opponents_sorted_i[0]])
        better_opponents_i = np.argwhere(opponents_scores[opponents_sorted_i] > player_score).flatten()
        if len(better_opponents_i) > 0:
            next_opponent_score = int(opponents_scores[opponents_sorted_i[better_opponents_i[0]]])
        else:
            next_opponent_score = best_opponent_score
        score_in_memory = int(self.game.current_player_score_in_memory)

        assert self.MIN_SCORE <= player_score <= self.MAX_SCORE
        assert self.MIN_SCORE <= best_opponent_score <= self.MAX_SCORE
        assert self.MIN_SCORE <= worst_opponent_score <= self.MAX_SCORE
        assert self.MIN_SCORE <= next_opponent_score <= self.MAX_SCORE
        assert self.MIN_SCORE <= score_in_memory <= self.MAX_SCORE

        return {
            'dice': np.append(self.game.current_player_kept_dice, self.game.current_player_new_dice).astype(np.int8) - 1,
            'n_locked_dice': len(self.game.current_player_kept_dice),
            'score_in_memory': score_in_memory - self.MIN_SCORE,
            'player_score': player_score - self.MIN_SCORE,
            'next_opponent_score': next_opponent_score - self.MIN_SCORE,
            'best_opponent_score': best_opponent_score - self.MIN_SCORE,
            'worst_opponent_score': worst_opponent_score - self.MIN_SCORE,
            'player_entered': np.array([self._get_current_player_entered()], np.int8),
            'any_opponent_entered': np.array([np.any(self._get_other_players_entered())])
        }

    def _let_opponent_act(self, idx):
        self.game.start_next_turn()
        agent = self.opponents_agents[idx]
        try:
            agent.act(self.game)
        except sim.GameException as ex:
            logging.warning(f'Opponent agent [{idx}] {str(agent)} tried to perform an illegal action: {ex}.')
        try:
            assert self.game.turn_finished
        except AssertionError:
            logging.warning(f'Opponent agent [{idx}] {str(agent)} did not finish their turn properly.')
        if self.game.is_over():
            logging.debug(f'Opponent agent [{idx}] {str(agent)} won the game after {self.game.n_rounds} rounds.')
            return True
        else:
            return False

    def _let_opponents_before_act(self):
        for idx in range(self.CONTROLLED_PLAYER_IDX):
            if self._let_opponent_act(idx):
                return True
        return False

    def _let_opponents_after_act(self):
        for idx in range(self.CONTROLLED_PLAYER_IDX + 1, self.game.n_players):
            if self._let_opponent_act(idx):
                return True
        return False

    def _handle_turn_end(self) -> bool:
        """ Return true if game's over """
        if self.game.is_over():
            return True
        else:
            if self._let_opponents_after_act():
                return True
            if self._let_opponents_before_act():
                return True
            self.game.start_next_turn()
            return False

    def _get_game_over_reward_or_penalty(self) -> int:
        if self.game.current_player_idx == self.CONTROLLED_PLAYER_IDX:
            logging.debug(f'Controlled agent won the game after {self.game.n_rounds} rounds.')
            time_reward = max(0, self.GAME_WON_REWARD_BASE - (self.game.n_rounds - 1) * self.GAME_WON_REWARD_DECREASE_RATE)
            return time_reward + self.GAME_WON_REWARD_PERSISTING_BONUS
        else:
            return self.OPPONENT_GAME_OVER_PENALTY

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
