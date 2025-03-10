import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_clusters(pca_data, cluster_labels):
    """
    Plot clusters using PCA-reduced data.

    Parameters:
        pca_data (pd.DataFrame): Data reduced to 2 principal components.
        cluster_labels (array-like): Cluster assignments.

    Returns:
        None (Displays plot)
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_data["PC1"], y=pca_data["PC2"], hue=cluster_labels, palette="viridis", alpha=0.8)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Clusters of Risk Perception")
    plt.legend(title="Cluster")
    plt.show()
