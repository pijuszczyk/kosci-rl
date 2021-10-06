from abc import abstractmethod
import numpy as np

import sim


class Agent:
    @abstractmethod
    def act(self, game: sim.Game):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError


class _InactiveAgent(Agent):
    def act(self, game: sim.Game):
        pass

    def __str__(self):
        return 'Inactive'


class _DummyAgent(Agent):
    def act(self, game: sim.Game):
        game.accept_roll()

    def __str__(self):
        return 'Dummy'


class _MinScoreAgent(Agent):
    def __init__(self, min_score: int):
        super().__init__()
        assert min_score >= 0
        self.min_score = min_score

    def act(self, game: sim.Game):
        if game.first_roll_failed:
            game.accept_roll()
            return
        score = sim.calculate_score(game.current_player_kept_dice, game.current_player_new_dice,
                                    game.current_player_score_in_memory)
        while score < self.min_score:
            if sim.can_reroll_every_dice(game.current_player_kept_dice, game.current_player_new_dice):
                game.do_full_reroll()
            else:
                to_keep = [i for i, die in enumerate(game.current_player_new_dice) if die == 1 or die == 5]
                if len(to_keep) == 0:
                    # no 1s or 5s, maybe we have a trio or quaddro but whatever #TODO
                    game.accept_roll()
                    return
                if len(to_keep) == len(game.current_player_new_dice):
                    # something's probably wrong
                    game.accept_roll()
                    return
                game.do_partial_reroll(np.array(to_keep))
            if game.roll_failed:
                game.accept_roll()
                return
            score = sim.calculate_score(game.current_player_kept_dice, game.current_player_new_dice,
                                        game.current_player_score_in_memory)
        game.accept_roll()

    def __str__(self):
        return f'Min score ({self.min_score})'


class AgentFactory:
    @staticmethod
    def make_agent(strategy: str):
        if strategy == 'inactive':
            return _InactiveAgent()
        if strategy == 'dummy':
            return _DummyAgent()
        if strategy == 'niegardzący':
            return _MinScoreAgent(40)
        if strategy == 'feeling_lucky':
            return _MinScoreAgent(200)
        raise ValueError
