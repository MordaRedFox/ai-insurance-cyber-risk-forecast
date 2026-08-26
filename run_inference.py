"""
Демонстрационный скрипт инференса модели на примере нескольких компаний
"""

import sys
sys.path.append('.')
import pandas as pd
from src import Inference, load_dataframe


if __name__ == '__main__':
    model_path = 'models/checkpoints/best_model.pth'
    preprocessor_path = 'models/scaler.pkl'
    model_config = 'configs/model_config.yaml'
    raw_data_path = 'data/presentation/presentation_companies.csv'
    df_raw = load_dataframe(raw_data_path).head(10)
    inference = Inference(
        model_path=model_path, preprocessor_path=preprocessor_path,
        model_config=model_config, device='cpu')
    predictions = inference.predict(df_raw)
    result = pd.concat([df_raw, predictions], axis=1)
    print("Результаты инференса (первые 5 компаний)")
    cols_to_show = [
        'Годовая выручка (млн руб.)', 'Отрасль', 'T1', 'T2', 'T3']
    print(result[cols_to_show].head(5).to_string())
