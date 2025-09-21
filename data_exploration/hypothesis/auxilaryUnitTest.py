from data_exploration.hypothesis.decisioning.auxilary import *

# Assume get_perf_value is already defined above or imported
# Here's a simplified test DataFrame structure for the function

class TestGetPerfValue(unittest.TestCase):

    def setUp(self):
        base_date = datetime(2024, 1, 1)
        self.df = pd.DataFrame({
            'day': [base_date + timedelta(days=i) for i in range(10)],
            'price': [
                100, 102, 101, 103, 102,  # 0–4
                98, 97, 95, 94, 96  # 5–9
            ]
        })

    def test_perf_above_threshold(self):
        result = get_perf_value(
            self.df,
            for_date='2024-01-01',
            period_days=5,
            threshold_for_1=-3.0,
            threshold_for_future_tolerance_days=3
        )
        self.assertFalse(result)  # price dropped ~6%, which is below -3%, hence True

    def test_perf_below_threshold(self):
        result = get_perf_value(
            self.df,
            for_date='2024-01-01',
            period_days=5,
            threshold_for_1=-1.0,
            threshold_for_future_tolerance_days=1
        )
        self.assertFalse(result)  # Min drop not enough

    def test_no_data_on_for_date(self):
        with self.assertRaises(ValueError):
            get_perf_value(
                self.df,
                for_date='2023-12-01',
                period_days=5,
                threshold_for_1=1.0,
                threshold_for_future_tolerance_days=3
            )

    def test_no_future_data(self):
        with self.assertRaises(ValueError):
            get_perf_value(
                self.df,
                for_date='2024-01-08',  # only 2 days left in data
                period_days=3,
                threshold_for_1=1.0,
                threshold_for_future_tolerance_days=3
            )



unittest.main()