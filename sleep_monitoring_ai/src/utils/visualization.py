import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SleepDataVisualizer:
    """
    Visualization tools for sleep monitoring research.
    """
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
    
    def correlation_heatmap(self, output_path='correlation_heatmap.png'):
        """Create correlation heatmap of environmental factors and sleep metrics"""
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation between Environmental Factors and Sleep Performance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def environmental_impact_boxplot(self, output_path='environmental_impact.png'):
        """Visualize how environmental factors impact sleep quality"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Impact of Environmental Factors on Sleep Quality')
        
        env_factors = ['noise_level', 'temperature', 'light_intensity', 'air_quality_index']
        
        for i, factor in enumerate(env_factors):
            row, col = divmod(i, 2)
            sns.boxplot(x=pd.qcut(self.data[factor], q=4), 
                        y=self.data['sleep_quality_score'], 
                        ax=axes[row, col])
            axes[row, col].set_title(f'{factor.replace("_", " ").title()} vs Sleep Quality')
            axes[row, col].set_xlabel(f'{factor.replace("_", " ").title()} Quartiles')
            axes[row, col].set_ylabel('Sleep Quality Score')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    visualizer = SleepDataVisualizer(
        'c:\\Users\\patri\\OneDrive\\Desktop\\Proiect IRA\\sleep_monitoring_ai\\sleep_environment_data.csv'
    )
    visualizer.correlation_heatmap()
    visualizer.environmental_impact_boxplot()
