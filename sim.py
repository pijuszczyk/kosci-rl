import logging
from enum import Enum
from typing import Tuple

import numpy as np


def roll(n_dice: int) -> np.ndarray:
    return np.random.choice(np.arange(1, 7, dtype='int32'), n_dice)


def partial_reroll(dice: np.ndarray, kept_ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kept_nums = dice[kept_ind]
    to_reroll = len(dice) - len(kept_nums)
    new_nums = roll(to_reroll)
    return kept_nums, new_nums


def calculate_numbers_occurrences(kept_dice: np.ndarray, new_dice: np.ndarray):
    occs = np.empty(6)
    for i in range(6):  # todo maybe optimize
        occs[i] = np.count_nonzero(new_dice == i + 1)
    occs[0] += np.count_nonzero(kept_dice == 1)
    occs[4] += np.count_nonzero(kept_dice == 5)
    return occs


def calculate_score_from_occurrences(occs: np.ndarray):
    ones_score = {0: 0, 1: 10, 2: 20, 3: 100, 4: 200, 5: 500}[occs[0]]
    fives_score = {0: 0, 1: 5, 2: 10, 3: 50, 4: 100, 5: 200}[occs[4]]
    trios_quattros_score = 0
    for i in range(6):
        if i in [0, 4]:
            continue
        if occs[i] == 3:
            trios_quattros_score += (i + 1) * 10
        elif occs[i] == 4:
            trios_quattros_score += (i + 1) * 20
    return ones_score + fives_score + trios_quattros_score


def calculate_score(kept_dice: np.ndarray, new_dice: np.ndarray, score_in_memory: int = 0) -> int:
    occs = calculate_numbers_occurrences(kept_dice, new_dice)
    occs_score = calculate_score_from_occurrences(occs)
    return score_in_memory + occs_score


def can_player_enter(roll_score: int) -> bool:
    return roll_score >= 50


def is_score_acceptable(player_score: int, roll_score: int, enable_thresholds: bool = True) -> bool:
    if enable_thresholds:
        if 500 <= player_score < 700:
            if player_score + roll_score < 700:
                return False
        if 800 <= player_score < 900:
            if player_score + roll_score < 900:
                return False
    if player_score + roll_score > 1000:
        return False
    return True


def has_player_won(player_score: int) -> bool:
    return player_score == 1000


def can_reroll_every_dice(kept_dice: np.ndarray, new_dice: np.ndarray) -> bool:
    return len(new_dice) == 0 and len(kept_dice) > 0


class ErrorType(Enum):
    UNKNOWN = 0
    TURN_ALREADY_FINISHED = 1
    TURN_NOT_FINISHED_YET = 2
    SCORE_UNACCEPTABLE = 3
    PLAYER_NOT_ENTERED_YET = 4
    BAD_REROLL_CHOICE = 5


class GameException(RuntimeError):
    def __init__(self, error_type: ErrorType = ErrorType.UNKNOWN, error_msg: str = ''):
        RuntimeError.__init__(self, f'{error_type.name} error: {error_msg}')


class Game:
    """
    Vocab:
    roll - any dice roll, be it partial or full
    turn - one for each player in a round, consisting of rolls
    round - consisting of turns
    important dice - dice results that grant positive score
    kept dice - important dice player chose not to (or was not able to) reroll
    new dice - dice player chose to reroll
    player score - total number of points a player has
    latent score - intermediate results, points from currently rolled dice not fully belonging to the player yet
    score in memory - previous latent score that was saved for later after reaching the state of all dice being important
    """
    N_DICE = 5

    OVERTAKE_PENALTY = -50
    BAD_ROLL_PENALTY = -50

    # game simplification params
    ENABLE_THRESHOLDS = True
    REQUIRE_ENTERING = True

    def __init__(self, n_players: int):
        self.n_players = n_players
        self.players_entered = np.full(n_players, not self.REQUIRE_ENTERING)
        self.players_scores = np.zeros(n_players)
        self.current_player_idx = n_players-1
        self.current_player_kept_dice = np.empty(self.N_DICE)
        self.current_player_new_dice = np.empty(0)
        self.current_player_score_in_memory = 0
        self.turn_finished = True
        self.overtakes = np.full(n_players, False)
        self.first_roll_failed = False
        self.roll_failed = False
        self.n_rounds = 0

    def _handle_overtakes(self, score_to_add: int):
        for other_player_idx in range(self.n_players):
            if other_player_idx == self.current_player_idx:
                continue
            if not self.players_entered[other_player_idx]:
                continue
            if self.players_scores[other_player_idx] < self.players_scores[self.current_player_idx]:
                continue
            if self.players_scores[other_player_idx] < self.players_scores[self.current_player_idx] + score_to_add:
                self.players_scores[other_player_idx] += self.OVERTAKE_PENALTY
                self.overtakes[other_player_idx] = True

    def _increment_current_player_idx(self):
        if self.current_player_idx == self.n_players - 1:
            self.n_rounds += 1
            self.current_player_idx = 0
        else:
            self.current_player_idx += 1

    def _reset_member_variable_for_next_turn(self):
        self.current_player_score_in_memory = 0
        self.turn_finished = False
        self.overtakes = np.full(self.n_players, False)
        self.first_roll_failed = False
        self.roll_failed = False

    def _roll_dice_for_next_turn(self):
        self.current_player_kept_dice = np.empty(0)
        self.current_player_new_dice = roll(self.N_DICE)

    def _handle_potential_first_roll_fail(self):
        if self.players_entered[self.current_player_idx]:
            if calculate_score(self.current_player_kept_dice, self.current_player_new_dice) == 0:
                self.first_roll_failed = True

    def _verify_start_next_turn(self):
        if not self.turn_finished:
            raise GameException(ErrorType.TURN_NOT_FINISHED_YET, 'Previous turn still in progress')

    def _log_start_next_turn(self):
        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(
                f'Player [{self.current_player_idx}] started a new turn. '
                f'Dice: {str(self.current_player_new_dice)}. '
                f'Latent score: {self._get_latent_score()}.')

    def start_next_turn(self):
        self._verify_start_next_turn()
        self._increment_current_player_idx()
        self._roll_dice_for_next_turn()
        self._reset_member_variable_for_next_turn()
        self._handle_potential_first_roll_fail()
        self._log_start_next_turn()

    def _log_do_partial_reroll(self, kept_idx: np.ndarray):
        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(
                f'Player [{self.current_player_idx}] rerolled dice, keeping indices: {str(kept_idx)}. '
                f'Dice: {str(np.append(self.current_player_kept_dice, self.current_player_new_dice).astype(np.int16))}. '
                f'Latent score: {self._get_latent_score()}. '
                f'Score in memory: {self.current_player_score_in_memory}.')

    def _do_partial_reroll(self, kept_idx: np.ndarray):
        try:
            kept, rerolled = partial_reroll(self.current_player_new_dice, kept_idx)
        except Exception as ex:
            raise GameException(ErrorType.BAD_REROLL_CHOICE, f'Invalid reroll choice - {str(ex)}')
        self.current_player_kept_dice = np.append(self.current_player_kept_dice, kept)
        self.current_player_new_dice = rerolled
        self._log_do_partial_reroll(kept_idx)

    def _get_latent_score(self):
        return calculate_score(self.current_player_kept_dice, self.current_player_new_dice)

    def _verify_partial_reroll(self, kept_idx: np.ndarray):
        if self.turn_finished:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'Too late to partially reroll')
        if self.first_roll_failed:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'First roll failed, partial rerolls are not available')
        if self.roll_failed:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'Previous roll failed, partial rerolls are not available')
        if len(kept_idx) == 0:
            raise GameException(ErrorType.BAD_REROLL_CHOICE, 'Must select at least one die for a reroll')
        if len(np.unique(kept_idx)) < len(kept_idx):
            raise GameException(ErrorType.BAD_REROLL_CHOICE, 'Cannot select a die multiple times for a reroll')

    def do_partial_reroll(self, kept_idx: np.ndarray):
        self._verify_partial_reroll(kept_idx)
        score_without_kept = calculate_score(self.current_player_kept_dice, np.empty(0))
        score_with_kept = calculate_score(self.current_player_kept_dice, self.current_player_new_dice[kept_idx])
        if score_with_kept == score_without_kept:
            raise GameException(ErrorType.BAD_REROLL_CHOICE, 'Must leave something new worth points to commit a reroll')
        self._do_partial_reroll(kept_idx)
        score_with_rerolled = self._get_latent_score()
        if score_with_rerolled == score_with_kept:
            self.roll_failed = True

    def _transfer_latent_score_to_memory(self):
        self.current_player_score_in_memory = calculate_score(self.current_player_kept_dice,
                                                              self.current_player_new_dice,
                                                              self.current_player_score_in_memory)

    def _verify_full_reroll(self):
        if self.turn_finished:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'Too late to reroll')
        if self.first_roll_failed:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'First roll failed, rerolls are not available')
        if self.roll_failed:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'Previous roll failed, rerolls are not available')
        if not can_reroll_every_dice(self.current_player_kept_dice, self.current_player_new_dice):
            raise GameException(ErrorType.BAD_REROLL_CHOICE, 'Cannot do a full reroll yet, must have all dice locked')

    def _log_do_full_reroll(self):
        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(
                f'Player [{self.current_player_idx}] rerolled all dice. '
                f'Dice: {str(np.append(self.current_player_kept_dice, self.current_player_new_dice).astype(np.int16))}. '
                f'Latent score: {self._get_latent_score()}. '
                f'Score in memory: {self.current_player_score_in_memory}.')

    def do_full_reroll(self):
        self._verify_full_reroll()
        self._transfer_latent_score_to_memory()
        self.current_player_kept_dice = np.empty(0)
        self.current_player_new_dice = roll(self.N_DICE)
        if self._get_latent_score() == 0:
            self.roll_failed = True

    def _add_score(self, score_to_add: int):
        self._handle_overtakes(score_to_add)
        self.players_scores[self.current_player_idx] += score_to_add

    def _log_accept_roll(self):
        if logging.root.isEnabledFor(logging.DEBUG):
            logging.debug(
                f'Player [{self.current_player_idx}] accepted the roll. '
                f'Current score: {self.players_scores[self.current_player_idx]}. '
                f'In game: {self.players_entered[self.current_player_idx]}.')

    def _accept_successful_roll(self):
        score_to_add = calculate_score(self.current_player_kept_dice, self.current_player_new_dice,
                                       self.current_player_score_in_memory)
        if self.players_entered[self.current_player_idx]:
            if is_score_acceptable(self.players_scores[self.current_player_idx], score_to_add, self.ENABLE_THRESHOLDS):
                self._add_score(score_to_add)
        else:
            if can_player_enter(score_to_add):
                self.players_entered[self.current_player_idx] = True
        self._log_accept_roll()

    def accept_roll(self):
        if self.turn_finished:
            raise GameException(ErrorType.TURN_ALREADY_FINISHED, 'Too late to accept a roll')
        if self.first_roll_failed:
            self.players_scores[self.current_player_idx] += self.BAD_ROLL_PENALTY
        else:
            self._accept_successful_roll()
        self.turn_finished = True

    def is_turn_beginning(self):
        return len(self.current_player_kept_dice) == 0 and self.current_player_score_in_memory == 0

    def is_over(self):
        return has_player_won(self.players_scores[self.current_player_idx])
