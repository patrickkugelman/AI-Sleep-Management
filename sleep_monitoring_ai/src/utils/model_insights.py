import plotly.express as px
import plotly.graph_objs as go
import shap
import pandas as pd
import numpy as np

class ModelInsights:
    @staticmethod
    def feature_importance_plot(feature_importance):
        """Create an interactive feature importance plot"""
        fig = px.bar(
            feature_importance, 
            x='feature', 
            y='importance',
            title='Feature Importance in Sleep Performance Prediction',
            labels={'importance': 'Importance Score', 'feature': 'Environmental Factors'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        return fig

    @staticmethod
    def performance_comparison_plot(models_dict):
        """Create a comparative performance visualization"""
        performance_data = []
        for model_name, model_info in models_dict.items():
            performance = model_info.get('performance', {})
            performance_data.append({
                'Model': model_name,
                'MSE': performance.get('MSE', 0),
                'MAE': performance.get('MAE', 0),
                'R2': performance.get('R2', 0)
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        # Melt for easier plotting
        df_melted = df_performance.melt(id_vars=['Model'], var_name='Metric', value_name='Value')
        
        fig = px.bar(
            df_melted, 
            x='Model', 
            y='Value', 
            color='Metric',
            title='Model Performance Comparison',
            barmode='group'
        )
        return fig

    @staticmethod
    def shap_summary_plot(shap_values, features):
        """Create a SHAP summary plot for model interpretability"""
        shap.summary_plot(
            shap_values, 
            features, 
            plot_type="bar", 
            show=False
        )
        return plt.gcf()

    def generate_comprehensive_report(self, predictor):
        """Generate a comprehensive model insights report"""
        report = {
            'feature_importance_plot': self.feature_importance_plot(predictor.models['random_forest']['feature_importance']),
            'performance_comparison_plot': self.performance_comparison_plot(predictor.models),
            'model_details': {
                name: {
                    'performance': model_info.get('performance', {}),
                    'feature_importance': model_info.get('feature_importance', [])
                } for name, model_info in predictor.models.items()
            }
        }
        return report
