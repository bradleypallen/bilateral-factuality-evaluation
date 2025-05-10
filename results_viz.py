import json, glob, re, pycm, pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, scipy.stats as stats
from IPython.display import display, Markdown

def display_cms(cms):
    fig = plt.figure(figsize=(20,14))
    gs = fig.add_gridspec(4, 5, hspace=0.5)
    axes = gs.subplots()
    for ax, (name, cm) in zip(axes.flat, cms):
        df = pd.DataFrame(cm.matrix).T.fillna(0)
        sns.heatmap(df, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
        ax.set_title(name, wrap=True, fontsize=9)
        ax.set(xlabel='Predicted', ylabel='Actual')
    for ax in axes.flat[len(cms):]:
        ax.set_visible(False)
    plt.show()

def unilateral_truth_value_distribution(df, n):
    model_evals = df.groupby('model_name')['evaluation'].value_counts().unstack(fill_value=0)
    for stat in ["t","f","n"]:
        if stat not in model_evals:
            model_evals[stat] = 0
    model_evals["t"] = model_evals["t"] / float(n)
    model_evals["n"] = model_evals["n"] / float(n)
    model_evals["f"] = model_evals["f"] / float(n)
    return model_evals[["t", "n", "f"]]

def bilateral_truth_value_distribution(df, n):
    model_evals = df.groupby('model_name')['evaluation'].value_counts().unstack(fill_value=0)
    for stat in ["t","f","n", "b"]:
        if stat not in model_evals:
            model_evals[stat] = 0
    model_evals["t"] = model_evals["t"] / float(n)
    model_evals["b"] = model_evals["b"] / float(n)
    model_evals["n"] = model_evals["n"] / float(n)
    model_evals["f"] = model_evals["f"] / float(n)
    return model_evals[["t", "n", "b",  "f"]]

def plot_metric_comparison(df, n, n_samples, metric="ACC", ymin=0.4, ymax=0.8, figsize=(15, 6)):
    # Set the colors
    unilateral_color = '#8884d8'
    bilateral_color = '#82ca9d'
    # Create figure with larger size
    plt.figure(figsize=figsize)
    # Calculate bar positions
    x = np.arange(len(df))
    width = 0.35
    # Create bars
    bars1 = plt.bar(x - width/2, df[f'unilateral {metric}'], width, label='Unilateral', color=unilateral_color, alpha=0.8)
    bars2 = plt.bar(x + width/2, df[f'bilateral {metric}'], width, label='Bilateral', color=bilateral_color, alpha=0.8)
    # Set y-axis limits
    plt.ylim(ymin, ymax)
    # Add value labels on top of bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.3f}', ha='center', va='bottom')
    autolabel(bars1)
    autolabel(bars2)
    # Customize the plot
    plt.ylabel(f'{metric} (given attempted)', fontsize=12)
    plt.title(f'Unilateral vs Bilateral Evaluation (N={int(n):d}, samples/evaluation={n_samples:d})', fontsize=14, pad=20)
    plt.xticks(x, df['model'], rotation=45, ha='right')
    # Adjust layout to prevent label cutoff
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_metrics_boxplot(df):
    # Create long-format data for the three metrics (ACC, AUC, F1 macro)
    unilateral_data = pd.DataFrame({
        'Metric': ['ACC'] * len(df) + ['AUC'] * len(df) + ['F1 macro'] * len(df),
        'Value': df['unilateral ACC'].tolist() + df['unilateral AUC'].tolist() + df['unilateral F1 macro'].tolist(),
        'Type': ['Unilateral'] * (len(df) * 3)
        })
    bilateral_data = pd.DataFrame({
        'Metric': ['ACC'] * len(df) + ['AUC'] * len(df) + ['F1 macro'] * len(df),
        'Value': df['bilateral ACC'].tolist() + df['bilateral AUC'].tolist() + df['bilateral F1 macro'].tolist(),
        'Type': ['Bilateral'] * (len(df) * 3)
    })
    # Combine the data
    plot_data = pd.concat([unilateral_data, bilateral_data])
    # Create the boxplot
    plt.figure(figsize=(10, 4))
    sns.boxplot(x='Metric', y='Value', hue='Type', data=plot_data, palette={'Unilateral': '#8884d8', 'Bilateral': '#82ca9d'})
    plt.title('Comparison of Unilateral vs Bilateral Performance Metrics (given attempted)')
    plt.ylabel('Score')
    plt.ylim(0.1, 1.0)
    plt.show()

def plot_tv_distributions(df):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    # Set width of bars
    barWidth = 0.2
    positions = np.arange(len(df))
    # Create bars
    plt.bar(positions - barWidth*1.5, df['t'], width=barWidth, color='#2ecc71', label='t')
    plt.bar(positions - barWidth*0.5, df['n'], width=barWidth, color='#3498db', label='n')
    plt.bar(positions + barWidth*0.5, df['b'], width=barWidth, color='#f39c12', label='b')
    plt.bar(positions + barWidth*1.5, df['f'], width=barWidth, color='#e74c3c', label='f')
    # Add labels, title and legend
    # plt.xlabel('model', fontsize=12)
    plt.ylabel('percentage occurence of truth value', fontsize=12)
    plt.title('Distribution of truth values across models', fontsize=14)
    plt.xticks(positions, df.index.tolist(), rotation=45, ha='right')
    plt.legend()
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Ensure layout is tight
    plt.tight_layout()
    # Add value annotations on top of each bar
    for i, value in enumerate(df['t']):
        plt.text(i - barWidth*1.5, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(df['n']):
        plt.text(i - barWidth*0.5, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(df['b']):
        plt.text(i + barWidth*0.5, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    for i, value in enumerate(df['f']):
        plt.text(i + barWidth*1.5, value + 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)
    plt.show()

def risk_coverage_plot(df):
    # Calculate risk as 1 - accuracy
    df['unilateral risk'] = 1 - df['unilateral ACC']
    df['bilateral risk'] = 1 - df['bilateral ACC']
    # Create a figure with a specific size
    plt.figure(figsize=(10, 10))
    # Plot the models
    plt.scatter(df['unilateral coverage'], df['unilateral risk'], 
        color='purple', alpha=0.5, marker='o', s=50, label='unilateral')
    plt.scatter(df['bilateral coverage'], df['bilateral risk'], 
        color='green', alpha=0.7, marker='D', s=70, label='bilateral')
    # Draw dashed lines connecting the same model in different methods
    for i in range(len(df)):
        model = df['model'][i]
        x_coords = [df['unilateral coverage'][i], df['bilateral coverage'][i]]
        y_coords = [df['unilateral risk'][i], df['bilateral risk'][i]]
        plt.plot(x_coords, y_coords, 'orange', linestyle='--', alpha=0.7)    
        plt.annotate(model, 
            (df['bilateral coverage'][i], df['bilateral risk'][i]),
                xytext=(-40, 10), textcoords='offset points')
    # Set labels and title
    plt.xlabel('coverage', fontsize=12)
    plt.ylabel('risk (1-accuracy)', fontsize=12)
    # Set axis limits
    plt.xlim(0.0, 1.02)
    plt.ylim(0.15, 0.6)
    # Add grid
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    # Add legend
    plt.legend(loc='lower left')
    # Show the plot
    plt.tight_layout()
    plt.show()

def load_run(run, n):
    cms = {}
    for file in glob.glob(f"experiments/{run}/unilateral/*.json"):
        model = re.match(r'.*/unilateral/(.*)-simpleqa.json', file).group(1)
        if model not in cms:
            cms[model] = {}
        df = pd.DataFrame.from_records(json.load(open(file, "r"))[:n])
        cms[model]["unilateral"] = pycm.ConfusionMatrix(df["label"].tolist(), df["evaluation"].tolist(), digit=2, classes=[ 't', 'f' ])
    for file in glob.glob(f"experiments/{run}/bilateral/*.json"):
        model = re.match(r'.*/bilateral/(.*)-simpleqa.json', file).group(1)
        if model not in cms:
            cms[model] = {}
        df = pd.DataFrame.from_records(json.load(open(file, "r"))[:n])
        cms[model]["bilateral"] = pycm.ConfusionMatrix(df["label"].tolist(), df["evaluation"].tolist(), digit=2, classes=[ 't', 'f' ])
    return cms

def generate_stats_df_from_cms(cms, n):
    stats_df = pd.DataFrame([ 
        {
            'model': model,
            'unilateral N': cms[model]["unilateral"].POP['t'], 
            'unilateral ACC': cms[model]["unilateral"].ACC_Macro, 
            'unilateral AUC': cms[model]["unilateral"].AUC['t'], 
            'unilateral F1 macro': cms[model]["unilateral"].F1_Macro,
            'bilateral N': cms[model]["bilateral"].POP['t'], 
            'bilateral ACC': cms[model]["bilateral"].ACC_Macro, 
            'bilateral AUC': cms[model]["bilateral"].AUC['t'], 
            'bilateral F1 macro': cms[model]["bilateral"].F1_Macro,
        } 
        for model in cms if "bilateral" in cms[model]
    ])
    stats_df['delta F1'] = stats_df['bilateral F1 macro'] - stats_df['unilateral F1 macro']
    stats_df['delta ACC'] = stats_df['bilateral ACC'] - stats_df['unilateral ACC']
    stats_df['unilateral coverage'] = stats_df['unilateral N'].apply(lambda x: x / n)
    stats_df['bilateral coverage'] = stats_df['bilateral N'].apply(lambda x: x / n)
    stats_df = stats_df[[ 'model', 'unilateral coverage', 'unilateral ACC', 'unilateral AUC', 'unilateral F1 macro', 
                     'bilateral coverage', 'bilateral ACC', 'bilateral AUC', 'bilateral F1 macro', 
                     'delta ACC', 'delta F1' ]].sort_values('delta ACC', ascending=False)
    return stats_df

def style_stats_df(stats_df):
    columns = [
        ('', 'model'),
        ('unilateral', 'coverage'),
        ('unilateral', 'ACC'),
        ('unilateral', 'AUC'),
        ('unilateral', 'F1 macro'),
        ('bilateral', 'coverage'),
        ('bilateral', 'ACC'),
        ('bilateral', 'AUC'),
        ('bilateral', 'F1 macro'),
        ('', 'delta ACC'),
        ('', 'delta F1')
    ]
    original_columns = ['model', 'unilateral coverage', 'unilateral ACC', 'unilateral AUC', 'unilateral F1 macro',
                   'bilateral coverage', 'bilateral ACC', 'bilateral AUC', 'bilateral F1 macro',
                   'delta ACC', 'delta F1']
    df_new = stats_df[original_columns].copy()
    df_new.columns = pd.MultiIndex.from_tuples(columns)
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.style.format({
        ('', 'model'): '{}',
        ('unilateral', 'coverage'): '{:.3f}',
        ('unilateral', 'ACC'): '{:.3f}',
        ('unilateral', 'AUC'): '{:.3f}',
        ('unilateral', 'F1 macro'): '{:.3f}',
        ('bilateral', 'coverage'): '{:.3f}',
        ('bilateral', 'ACC'): '{:.3f}',
        ('bilateral', 'AUC'): '{:.3f}',
        ('bilateral', 'F1 macro'): '{:.3f}',
        ('', 'delta ACC'): '{:.3f}',
        ('', 'delta F1'): '{:.3f}'
    })
    return df_new

