"""
Конфигурационный файл с параметрами для AutoML пайплайна
"""


# Параметры предобработки данных
PREPROCESSING_CONFIG = {
    'random_state': 42,  # Seed для воспроизводимости
    'categorical_features': ['feature_1', 'feature_2', 'feature_3'],  # Категориальные признаки
    'numerical_features': [f'feature_{i}' for i in range(4, 21)]  # Числовые признаки
    # Примечание: Разделение на train/test происходит в пропорциях:
    # - 80% для train
    # - 20% для test
    # Дубликаты помечаются отдельно, один остается в train, остальные как duplicate
}

# Параметры отбора признаков
FEATURE_SELECTION_CONFIG = {
    'random_state': 42,  # Seed для воспроизводимости
    'n_trials': 50,  # Количество попыток для Optuna
    'correlation_threshold': 1.0,  # Порог корреляции для удаления признаков
    'min_features_fi': 25,  # Минимальное количество признаков для feature importance
    'min_features_pi': 20,  # Минимальное количество признаков для permutation importance
    'stop_threshold': 0.02  # Порог относительного изменения ROC-AUC для остановки
}

# Параметры моделей
MODEL_CONFIG = {
    'random_state': 42,
    'n_trials': 20,  # Уменьшаем количество попыток для быстрого поиска
    
    # Параметры CatBoost
    'catboost': {
        'iterations': (100, 300),  # Уменьшаем максимальное количество итераций
        'learning_rate': (0.01, 0.3),
        'depth': (4, 8),  # Уменьшаем максимальную глубину
        'l2_leaf_reg': (1, 10),
        'bootstrap_type': ['Bernoulli'],
        'subsample': (0.6, 1.0),
        'random_strength': (1, 10),
        'od_type': ['Iter'],
        'od_wait': (10, 30)  # Уменьшаем время ожидания
    },
    
    # Параметры XGBoost
    'xgboost': {
        'max_depth': (3, 7),  # Уменьшаем максимальную глубину
        'learning_rate': (0.01, 0.3),
        'n_estimators': (100, 300),  # Уменьшаем максимальное количество деревьев
        'subsample': (0.6, 1.0),
        'reg_lambda': (0, 5)
    },
    
    # Параметры LightGBM
    'lightgbm': {
        'max_depth': (3, 7),  # Уменьшаем максимальную глубину
        'learning_rate': (0.01, 0.3),
        'n_estimators': (100, 300),  # Уменьшаем максимальное количество деревьев
        'num_leaves': (31, 63),  # Уменьшаем максимальное количество листьев
        'subsample': (0.6, 1.0),
        'min_child_samples': (5, 50),  # Уменьшаем максимальное значение
        'reg_lambda': (0, 5)
    }
}
# Параметры оптимизации порога
THRESHOLD_CONFIG = {
    'metrics': ['f1', 'recall'],
    'target_recall': 0.8  # Целевой уровень recall для дефолтных клиентов
}

# Параметры интерпретации
EXPLAINABILITY_CONFIG = {
    'random_state': 42,
    'sample_size': 1000,  # Размер выборки для расчета SHAP значений
    'top_features': 10,  # Количество топовых признаков для анализа взаимодействий
    'num_features': 10,  # Количество признаков для отображения в LIME
    'class_names': ['Хороший', 'Плохой'],  # Названия классов для LIME
    'mode': 'classification'  # Режим работы LIME
} 