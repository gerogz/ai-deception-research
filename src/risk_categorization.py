import pandas as pd

# Expected risk levels per scenario
expected_risk_levels = {
    'Obstetrician': 'high',
    'App': 'medium',
    'Waiter': 'low',
    'Birthday': 'low',
    'Drawing': 'medium',
    'Security Guard': 'high'
}

# Risk thresholds
risk_thresholds = {
    'low': (0, 33),
    'medium': (34, 66),
    'high': (67, 100)
}

# Scenario mappings
scenario_mapping = {
    'obst': 'Obstetrician',
    'app': 'App',
    'wait': 'Waiter',
    'birth': 'Birthday',
    'teach': 'Drawing',
    'sec': 'Security Guard'
}

scenario_types = {
    'h_o': 'Human Omission',
    'h_c': 'Human Commission',
    'r_o': 'Robot Omission',
    'r_c': 'Robot Commission'
}

def categorize_risk(value):
    """
    Categorize a given numerical risk value based on predefined thresholds.
    
    Parameters:
        value (float): Risk score (0-100).
    
    Returns:
        str: Risk category ('low', 'medium', 'high', or 'undefined').
    """
    for level, (low, high) in risk_thresholds.items():
        if pd.notnull(value) and low <= value <= high:
            return level
    return 'undefined'

def compute_risk_levels(data_cleaned):
    """
    Compute average risk levels, categorize them, and compare with expected values.

    Parameters:
        data_cleaned (pd.DataFrame): Preprocessed DataFrame with risk values.
    
    Returns:
        pd.DataFrame: Summary DataFrame with computed risk levels.
    """
    risk_summary = {}

    for scenario_prefix, scenario_name in scenario_mapping.items():
        for scenario_suffix, scenario_type in scenario_types.items():
            self_risk_col = f'{scenario_prefix}_{scenario_suffix}_selfRisk_1'
            other_risk_col = f'{scenario_prefix}_{scenario_suffix}_otherRisk_1'

            if self_risk_col in data_cleaned.columns or other_risk_col in data_cleaned.columns:
                available_columns = [col for col in [self_risk_col, other_risk_col] if col in data_cleaned.columns]
                scenario_data = data_cleaned[available_columns].dropna(how='all')

                avg_self_risk = scenario_data[self_risk_col].mean() if self_risk_col in data_cleaned.columns else None
                avg_other_risk = scenario_data[other_risk_col].mean() if other_risk_col in data_cleaned.columns else None

                self_risk_category = categorize_risk(avg_self_risk) if avg_self_risk is not None else 'No data'
                other_risk_category = categorize_risk(avg_other_risk) if avg_other_risk is not None else 'No data'

                risk_summary[f'{scenario_name} ({scenario_type})'] = {
                    'Average SelfRisk': avg_self_risk if avg_self_risk is not None else 'No data',
                    'SelfRisk Category': self_risk_category,
                    'Average OtherRisk': avg_other_risk if avg_other_risk is not None else 'No data',
                    'OtherRisk Category': other_risk_category,
                    'Expected Risk': expected_risk_levels[scenario_name],
                    'SelfRisk Matches': self_risk_category == expected_risk_levels[scenario_name] if avg_self_risk is not None else 'No data',
                    'OtherRisk Matches': other_risk_category == expected_risk_levels[scenario_name] if avg_other_risk is not None else 'No data'
                }

    return pd.DataFrame(risk_summary).T
