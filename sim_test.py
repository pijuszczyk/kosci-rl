import unittest
import sim
import numpy as np


class RollTests(unittest.TestCase):
    REQUIRED_ATTEMPTS = 20  # to test in various random conditions

    def test_roll_n_returns_n_dice(self):
        for i in range(100):
            self.assertEqual(len(sim.roll(i)), i)

    def test_roll_returns_valid_nums(self):
        for _ in range(self.REQUIRED_ATTEMPTS):
            dice = sim.roll(5)
            self.assertTrue(np.all(dice >= 1))
            self.assertTrue(np.all(dice <= 6))
            self.assertEqual(dice.dtype, 'int32')

    def test_partial_reroll_keeps_elements(self):
        for _ in range(self.REQUIRED_ATTEMPTS):
            for n_all in range(1, 6):
                dice = sim.roll(n_all)
                for n_kept in [1, n_all]:
                    kept_idx = np.random.choice(np.arange(n_all), n_kept, replace=False)
                    kept_nums = dice[kept_idx]
                    ret_kept_nums, _ = sim.partial_reroll(dice, kept_idx)
                    self.assertTrue(np.all(ret_kept_nums == kept_nums))


class ScoringTests(unittest.TestCase):
    def test_calculate_score_nothing(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([2, 4, 3, 6, 2])), 0)
        self.assertEqual(sim.calculate_score(np.array([4, 3]), np.array([2, 4, 2])), 0)

    def test_calculate_score_1x1(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([4, 1, 2, 3, 2])), 10)
        self.assertEqual(sim.calculate_score(np.array([1]), np.array([2, 3, 3, 2])), 10)

    def test_calculate_score_2x1(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([1, 6, 1, 2, 2])), 20)
        self.assertEqual(sim.calculate_score(np.array([2, 1]), np.array([1, 6, 2])), 20)

    def test_calculate_score_3x1(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([1, 6, 4, 1, 1])), 100)
        self.assertEqual(sim.calculate_score(np.array([2, 3]), np.array([1, 1, 1])), 100)

    def test_calculate_score_4x1(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([1, 1, 4, 1, 1])), 200)
        self.assertEqual(sim.calculate_score(np.array([1]), np.array([1, 3, 1, 1])), 200)

    def test_calculate_score_5x1(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([1, 1, 1, 1, 1])), 500)
        self.assertEqual(sim.calculate_score(np.array([1]), np.array([1, 1, 1, 1])), 500)

    def test_calculate_score_1x5(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([4, 5, 2, 3, 2])), 5)
        self.assertEqual(sim.calculate_score(np.array([5]), np.array([2, 3, 3, 2])), 5)

    def test_calculate_score_2x5(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([5, 6, 3, 5, 2])), 10)
        self.assertEqual(sim.calculate_score(np.array([5, 6]), np.array([5, 6, 2])), 10)

    def test_calculate_score_3x5(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([2, 3, 5, 5, 5])), 50)
        self.assertEqual(sim.calculate_score(np.array([2, 5]), np.array([6, 5, 5])), 50)

    def test_calculate_score_4x5(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([3, 5, 5, 5, 5])), 100)
        self.assertEqual(sim.calculate_score(np.array([5]), np.array([5, 2, 5, 5])), 100)

    def test_calculate_score_5x5(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([5, 5, 5, 5, 5])), 200)
        self.assertEqual(sim.calculate_score(np.array([5, 5, 5]), np.array([5, 5])), 200)

    def test_calculate_score_1_and_5(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([1, 3, 4, 5, 6])), 15)
        self.assertEqual(sim.calculate_score(np.array([2, 1]), np.array([4, 5, 4])), 15)

    def test_calculate_score_3x2(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([3, 2, 2, 4, 2])), 20)
        self.assertEqual(sim.calculate_score(np.array([2, 2, 4, 2]), np.array([3])), 0)
        self.assertEqual(sim.calculate_score(np.array([2]), np.array([2, 2, 3, 4])), 0)

    def test_calculate_score_4x2(self):
        self.assertEqual(sim.calculate_score(np.array([]), np.array([3, 2, 2, 2, 2])), 40)
        self.assertEqual(sim.calculate_score(np.array([2, 2, 2, 2]), np.array([6])), 0)
        self.assertEqual(sim.calculate_score(np.array([2, 2]), np.array([2, 4, 2])), 0)

    def test_calculate_score_invalid_nums(self):
        self.assertEqual(sim.calculate_score(np.array([-1, 7, 0]), np.array([2, 5])), 5)

    def test_calculate_score_invalid_dice_num_1(self):
        self.assertRaises(KeyError, lambda: sim.calculate_score(np.array([1, 1, 1, 1, 1, 1, 1, 1]), np.array([2, 1, 3, 4, 1, 1])))
        self.assertEqual(sim.calculate_score(np.array([1]), np.array([])), 10)  # todo maybe it could also raise

    def test_calculate_score_invalid_dice_num_trio_quattro(self):
        self.assertEqual(sim.calculate_score(np.array([2, 2, 2, 2, 2, 2, 2]), np.array([])), 0)  # todo maybe it could also raise


class ConditionsTests(unittest.TestCase):
    def test_roll_0_no_enter(self):
        self.assertFalse(sim.can_player_enter(0))

    def test_roll_45_no_enter(self):
        self.assertFalse(sim.can_player_enter(45))

    def test_roll_50_enter(self):
        self.assertTrue(sim.can_player_enter(50))

    def test_roll_200_enter(self):
        self.assertTrue(sim.can_player_enter(200))

    def test_roll_neg_1_no_enter(self):
        self.assertFalse(sim.can_player_enter(-1))

    def test_first_threshold_from_start_to_end_exit(self):
        self.assertTrue(sim.is_score_acceptable(500, 200))

    def test_first_threshold_from_start_not_enough_no_exit(self):
        self.assertFalse(sim.is_score_acceptable(500, 195))

    def test_first_threshold_from_start_lots_exit(self):
        self.assertTrue(sim.is_score_acceptable(500, 205))

    def test_first_threshold_from_middle_enough_exit(self):
        self.assertTrue(sim.is_score_acceptable(605, 100))

    def test_second_threshold_from_start_to_end_exit(self):
        self.assertTrue(sim.is_score_acceptable(800, 100))

    def test_second_threshold_from_start_not_enough_no_exit(self):
        self.assertFalse(sim.is_score_acceptable(800, 95))

    def test_second_threshold_from_start_lots_exit(self):
        self.assertTrue(sim.is_score_acceptable(800, 150))

    def test_second_threshold_from_middle_enough_exit(self):
        self.assertTrue(sim.is_score_acceptable(850, 70))

    def test_first_threshold_through_second_exit(self):
        self.assertTrue(sim.is_score_acceptable(520, 390))

    def test_first_threshold_to_second_exit_kinda(self):
        self.assertTrue(sim.is_score_acceptable(530, 300))

    def test_before_first_threshold_into_first(self):
        self.assertTrue(sim.is_score_acceptable(400, 120))

    def test_before_first_threshold(self):
        self.assertTrue(sim.is_score_acceptable(200, 100))

    def test_between_thresholds(self):
        self.assertTrue(sim.is_score_acceptable(750, 30))

    def test_after_thresholds(self):
        self.assertTrue(sim.is_score_acceptable(900, 50))

    def test_over_1000_points_fail(self):
        self.assertFalse(sim.is_score_acceptable(950, 100))

    def test_exactly_1000_points_ok(self):
        self.assertTrue(sim.is_score_acceptable(970, 30))

    def test_score_below_1000_not_won(self):
        self.assertFalse(sim.has_player_won(440))

    def test_score_exactly_1000_won(self):
        self.assertTrue(sim.has_player_won(1000))

    def test_score_above_1000_fail(self):
        self.assertFalse(sim.has_player_won(1200))  # todo maybe raise

    def test_reroll_all_kept_allow(self):
        self.assertTrue(sim.can_reroll_every_dice(np.array([2, 3, 1, 4, 5]), np.array([])))

    def test_reroll_all_new_disallow(self):
        self.assertFalse(sim.can_reroll_every_dice(np.array([]), np.array([1, 2, 3, 4, 5])))

    def test_reroll_some_new_disallow(self):
        self.assertFalse(sim.can_reroll_every_dice(np.array([6, 5]), np.array([4, 2, 1])))

    def test_reroll_no_dice_disallow(self):
        self.assertFalse(sim.can_reroll_every_dice(np.array([]), np.array([])))

    def test_reroll_invalid_dice_num(self):
        self.assertTrue(sim.can_reroll_every_dice(np.array([2, 3, 2, 5, 4, 3, 1, 1, 6, 5, 2]), np.array([])))


if __name__ == '__main__':
    unittest.main()
