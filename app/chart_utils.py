from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional, Union
from bson import ObjectId
from collections import defaultdict

class ChartDataProcessor:
    @staticmethod
    def validate_date_range(start_date, end_date):
        try:
            if not start_date or not end_date:
                return None, None
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            if start > end:
                start, end = end, start
            return start, end
        except ValueError:
            return None, None

    @staticmethod
    def process_daily_trend(data):
        if not data:
            return {'labels': [], 'data': [{'label': 'Damage Detected', 'data': []}, {'label': 'No Damage', 'data': []}]}

        # Sort data by date
        sorted_data = sorted(data, key=lambda x: x['_id']['date'])

        # Since dates are already formatted by MongoDB's $dateToString, use them directly
        labels = [item['_id']['date'] for item in sorted_data]
        
        # Initialize data structures
        damage_detected = defaultdict(int)
        no_damage = defaultdict(int)

        # Process the data
        for item in sorted_data:
            date = item['_id']['date']
            if item['_id']['success']:
                damage_detected[date] += item['count']
            else:
                no_damage[date] += item['count']

        # Create the final data structure
        unique_dates = sorted(set(labels))
        damage_data = [damage_detected[date] for date in unique_dates]
        no_damage_data = [no_damage[date] for date in unique_dates]

        return {
            'labels': unique_dates,
            'data': [
                {'label': 'Damage Detected', 'data': damage_data},
                {'label': 'No Damage', 'data': no_damage_data}
            ]
        }

    @staticmethod
    def process_damage_distribution(data):
        if not data:
            return {'labels': [], 'data': []}
            
        labels = []
        values = []
        
        for item in data:
            damage_type = item['_id'] if item['_id'] else 'Unknown'
            labels.append(damage_type)
            values.append(item['count'])
            
        return {'labels': labels, 'data': values}

    @staticmethod
    def process_processing_times(data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate processing time statistics with outlier handling."""
        if not data:
            return {'avg_time': 0, 'max_time': 0, 'min_time': 0}

        # Extract valid processing times
        times = [float(item['processing_time']) for item in data 
                if isinstance(item.get('processing_time'), (int, float))]

        if not times:
            return {'avg_time': 0, 'max_time': 0, 'min_time': 0}

        # Handle outliers using IQR method
        q1 = np.percentile(times, 25)
        q3 = np.percentile(times, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        filtered_times = [t for t in times if lower_bound <= t <= upper_bound]

        if not filtered_times:
            filtered_times = times  # Use original data if filtering removes all values

        return {
            'avg_time': np.mean(filtered_times),
            'max_time': max(filtered_times),
            'min_time': min(filtered_times)
        }

    @staticmethod
    def process_model_accuracy(data):
        if not data:
            return {}
            
        accuracy_data = {}
        for item in data:
            model_type = item['_id'] if item['_id'] else 'Unknown'
            total = item['true_positives'] + item['false_positives']
            accuracy = (item['true_positives'] / total * 100) if total > 0 else 0
            accuracy_data[model_type] = {
                'accuracy': round(accuracy, 2),
                'total_predictions': total,
                'true_positives': item['true_positives'],
                'false_positives': item['false_positives']
            }
            
        return accuracy_data

    @staticmethod
    def process_error_types(data):
        if not data:
            return {'labels': [], 'data': []}
            
        labels = []
        values = []
        
        for item in data:
            error_type = item['_id'] if item['_id'] else 'Unknown Error'
            labels.append(error_type)
            values.append(item['count'])
            
        return {'labels': labels, 'data': values}

    @staticmethod
    def process_confidence_trends(data):
        if not data:
            return {'data': {}}
            
        confidence_by_model = defaultdict(list)
        dates_by_model = defaultdict(list)
        
        for item in data:
            model = item['_id']['model'] if item['_id']['model'] else 'Unknown'
            confidence_by_model[model].append(item['avg_confidence'])
            dates_by_model[model].append(item['_id']['date'])
            
        result = {}
        for model in confidence_by_model:
            result[model] = {
                'dates': dates_by_model[model],
                'values': confidence_by_model[model]
            }
            
        return {'data': result}

    @staticmethod
    def validate_mongodb_id(id_str: str) -> bool:
        """Validate MongoDB ObjectId."""
        try:
            ObjectId(id_str)
            return True
        except:
            return False

    @staticmethod
    def sanitize_string(value: str) -> str:
        """Sanitize string inputs."""
        if not isinstance(value, str):
            return ''
        # Remove potential MongoDB operators and special characters
        return ''.join(c for c in value if c.isalnum() or c in ['-', '_', ' '])

    @staticmethod
    def validate_numeric(value: Any) -> Optional[float]:
        """Validate and convert numeric inputs."""
        try:
            return float(value)
        except (ValueError, TypeError):
            return None 