
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def generate_plots():
    # Setup directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    static_metrics_dir = os.path.join(base_dir, "static", "metrics")
    os.makedirs(static_metrics_dir, exist_ok=True)
    
    # 1. Retrieval Metrics Plot
    retrieval_csv_path = os.path.join(base_dir, "retrieval_metrics.csv")
    if os.path.exists(retrieval_csv_path):
        try:
            df_retrieval = pd.read_csv(retrieval_csv_path)
            
            # Melt for seaborn grouped bar plot
            df_melted = df_retrieval.melt(id_vars="Config", var_name="Metric", value_name="Score")
            
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(data=df_melted, x="Config", y="Score", hue="Metric", palette="viridis")
            
            plt.title("Search Performance (Hit Rate vs MRR)", fontsize=16)
            plt.ylim(0, 1.1)
            plt.ylabel("Score (0-1)", fontsize=12)
            plt.xlabel("Configuration", fontsize=12)
            
            # Add values on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')
                
            plt.tight_layout()
            output_path = os.path.join(static_metrics_dir, "retrieval_metrics.png")
            plt.savefig(output_path)
            print(f"✅ Generated: {output_path}")
            plt.close()
        except Exception as e:
            print(f"❌ Error processing retrieval metrics: {e}")
    else:
        print(f"⚠️ File not found: {retrieval_csv_path}")

    # 2. RAGAS Metrics Plot
    ragas_csv_path = os.path.join(base_dir, "ragas_metrics.csv")
    if os.path.exists(ragas_csv_path):
        try:
            df_ragas = pd.read_csv(ragas_csv_path)
            
            # Calculate Averages
            avg_faithfulness = df_ragas['faithfulness'].mean()
            avg_relevancy = df_ragas['answer_relevancy'].mean()
            
            metrics = ['Faithfulness', 'Answer Relevancy']
            scores = [avg_faithfulness, avg_relevancy]
            
            plt.figure(figsize=(8, 6))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(x=metrics, y=scores, palette="magma", width=0.5)
            
            plt.title("Generation Quality (RAGAS)", fontsize=16)
            plt.ylim(0, 1.1)
            plt.ylabel("Score (0-1)", fontsize=12)
            
            # Add values on top of bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f')
                
            plt.tight_layout()
            output_path = os.path.join(static_metrics_dir, "ragas_metrics.png")
            plt.savefig(output_path)
            print(f"✅ Generated: {output_path}")
            plt.close()
        except Exception as e:
            print(f"❌ Error processing RAGAS metrics: {e}")
    else:
        print(f"⚠️ File not found: {ragas_csv_path}")

if __name__ == "__main__":
    generate_plots()
