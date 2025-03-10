import pandas as pd

def style_matches(val):
    """Style the match results with colors (green for match, red for no match)."""
    if isinstance(val, bool):
        color = '#e8f5e9' if val else '#ffebee'  # Green for True, Red for False
        return f'background-color: {color}; text-align: center'
    return ''

def format_and_style(df):
    """
    Format and style the DataFrame for better presentation.

    Parameters:
        df (pd.DataFrame): DataFrame containing risk comparison results.
    
    Returns:
        pd.io.formats.style.Styler: Styled DataFrame.
    """
    return df.style\
        .applymap(style_matches, subset=['SelfRisk Matches', 'OtherRisk Matches'])\
        .set_properties(**{
            'text-align': 'left',
            'border': '1px solid #ddd',
            'padding': '8px'
        })\
        .set_caption('Risk Level Match Comparison')
