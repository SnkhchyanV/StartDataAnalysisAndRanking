import pandas as pd

class InvalidWeight(Exception):
    """Exception raised if invalid weight naming or value occurs."""

    def __init__(self, message, error_code=400):
        super().__init__(message)
        self.error_code = error_code
        self.message = message

    def __str__(self):
        return f"{self.message} (Error Code: {self.error_code})"

class StartupRatingTool:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.weights = {
            'Industry': 0,
            'Latest Funding Round': 0,
            'Latest Funding Date': 0,
            'Total Funding': 0,
            'All Investors': 0,
            'Latest Valuation': 0,
            'Latest Funding Amount': 0,
            'Exit Round': 0,
            'Exit Date': 0,
            'Acquirers': 0,
            'Mosaic (Overall)': 0,
            'Date Added': 0,
            'Applications': 0
        }
        self.columns_to_display = []

    def set_weights(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.weights:
                if value < 0:
                    raise InvalidWeight(f"Weight for {key} cannot be negative.")
                self.weights[key] = value
            else:
                raise InvalidWeight(f"The parameter '{key}' is not represented in data. Check the spelling or parameter list.")

    def rerank(self):
        # Reset the composite score
        self.data['Composite Score'] = 0
        self.columns_to_display = ['Companies', 'Composite Score']

        # Select parameters with non-zero weights
        for parameter, weight in self.weights.items():
            if weight != 0:
                score_feature = f'{parameter} Score'
                if score_feature in self.data.columns:
                    self.data['Composite Score'] += self.data[score_feature] * weight
                    self.columns_to_display.extend([parameter, score_feature])
                else:
                    print(f"Warning: '{score_feature}' column missing in data.")

    def display_ranking(self, sort_value='Composite Score', ascending=False, n_rows=10):
        if isinstance(sort_value, str):
            sort_value = [sort_value]
        ranking_data = self.data[self.columns_to_display].sort_values(by=sort_value, ascending=ascending)
        print(ranking_data.head(n_rows))