import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from database import FactorizationDB
import pandas as pd

def generate_statistical_plots(db_path="factorization_research_v2.db"):
    """
    Generate statistical charts for factorization function analysis
    """
    db = FactorizationDB()
    
    # Retrieve data from database
    conn = sqlite3.connect(db_path)
    query = """
        SELECT cf.name, tr.score, tr.precision, tr.recall, tr.computation_time, tr.efficiency_score
        FROM candidate_functions cf
        JOIN test_results tr ON cf.id = tr.function_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Create charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Factorization Function Statistical Analysis', fontsize=16, fontweight='bold')
    
    # 1. F1 Score Distribution
    axes[0, 0].hist(df['score'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('F1 Score Distribution')
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Precision vs Recall relationship
    axes[0, 1].scatter(df['precision'], df['recall'], alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)  # Equality line
    axes[0, 1].set_title('Precision vs Recall Relationship')
    axes[0, 1].set_xlabel('Precision')
    axes[0, 1].set_ylabel('Recall')
    
    # 3. Execution Time Distribution
    axes[0, 2].hist(df['computation_time'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Execution Time Distribution')
    axes[0, 2].set_xlabel('Execution Time (milliseconds)')
    axes[0, 2].set_ylabel('Count')
    
    # 4. Efficiency chart
    axes[1, 0].scatter(df['score'], df['efficiency_score'], alpha=0.6)
    axes[1, 0].set_title('Efficiency Chart (F1 vs Efficiency)')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_ylabel('Efficiency Score')
    
    # 5. Top 10 functions by F1
    top_functions = df.groupby('name')['score'].mean().nlargest(10)
    axes[1, 1].barh(range(len(top_functions)), top_functions.values)
    axes[1, 1].set_yticks(range(len(top_functions)))
    axes[1, 1].set_yticklabels(top_functions.index, fontsize=8)
    axes[1, 1].set_title('Top 10 Functions by F1')
    axes[1, 1].set_xlabel('Average F1')
    
    # 6. F1 vs Efficiency relationship
    axes[1, 2].scatter(df['score'], df['efficiency_score'], alpha=0.6)
    z = np.polyfit(df['score'], df['efficiency_score'], 1)
    p = np.poly1d(z)
    axes[1, 2].plot(df['score'], p(df['score']), "r--", alpha=0.8)
    axes[1, 2].set_title('F1 vs Efficiency Relationship')
    axes[1, 2].set_xlabel('F1 Score')
    axes[1, 2].set_ylabel('Efficiency Score')
    
    plt.tight_layout()
    plt.savefig('analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Charts saved to analysis_plots.png")
    
    # Display basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Number of measurements: {len(df)}")
    print(f"Average F1: {df['score'].mean():.3f}")
    print(f"Highest F1: {df['score'].max():.3f}")
    print(f"Average Precision: {df['precision'].mean():.3f}")
    print(f"Average Recall: {df['recall'].mean():.3f}")
    print(f"Average Execution Time: {df['computation_time'].mean():.3f} ms")
    
    # Top functions
    print("\n=== Top 5 Functions by F1 ===")
    top_funcs = df.groupby('name')['score'].mean().nlargest(5)
    for i, (func_name, avg_score) in enumerate(top_funcs.items(), 1):
        print(f"{i}. {func_name}: {avg_score:.3f}")
    
    plt.show()

def generate_detailed_analysis():
    """
    Generate detailed analysis of different functions
    """
    db = FactorizationDB()
    
    # Retrieve top functions
    best_functions = db.get_best_functions(min_score=0.0, limit=20)
    
    print("\n=== Detailed Analysis of Top Functions ===")
    for i, (name, code, avg_score, avg_precision, avg_recall, count, total_tp, total_fp) in enumerate(best_functions[:10], 1):
        print(f"\n{i}. {name}")
        print(f"   Average F1: {avg_score:.3f}")
        print(f"   Average Precision: {avg_precision:.3f}")
        print(f"   Average Recall: {avg_recall:.3f}")
        print(f"   Number of tests: {count}")
        print(f"   Total TP: {total_tp}, FP: {total_fp}")
        
        # Calculate FP rate (false positive rate)
        if total_tp + total_fp > 0:
            fpr = total_fp / (total_tp + total_fp)  # False Positive Rate
            print(f"   False Positive Rate: {fpr:.3f}")
    
    # Distribution analysis
    stats = db.get_statistics()
    print(f"\n=== Result Distribution ===")
    for category, count in stats['score_distribution']:
        print(f"{category}: {count} results")

if __name__ == "__main__":
    print("Generating statistical analysis...")
    generate_statistical_plots()
    generate_detailed_analysis()