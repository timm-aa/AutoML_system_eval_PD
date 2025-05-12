"""
Модуль для подбора гиперпараметров моделей
"""

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

def create_model(model_name, params=None):
    """
    Создание модели по имени
    
    Parameters:
    -----------
    model_name : str
        Название модели
    params : dict, optional
        Параметры модели
        
    Returns:
    --------
    model : object
        Обученная модель
    """
    if params is None:
        params = {}
        
    # Базовые параметры для каждой модели
    base_params = {
        'catboost': {
            'verbose': False,
            'random_state': 42,
            'eval_metric': 'AUC'
        },
        'xgboost': {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        },
        'lightgbm': {
            'objective': 'binary',
            'eval_metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': 42
        }
    }
    
    # Объединяем базовые параметры с оптимизированными
    model_params = {**base_params.get(model_name, {}), **params}
    
    models = {
        'catboost': CatBoostClassifier(**model_params),
        'xgboost': xgb.XGBClassifier(**model_params),
        'lightgbm': lgb.LGBMClassifier(**model_params)
    }
    
    return models.get(model_name)

def objective(trial, model_name, X_train, y_train, X_val, y_val, config):
    """
    Целевая функция для Optuna
    
    Parameters:
    -----------
    trial : optuna.Trial
        Объект trial
    model_name : str
        Название модели
    X_train, y_train : array-like
        Обучающие данные
    X_val, y_val : array-like
        Валидационные данные
    config : dict
        Конфигурация с параметрами моделей
        
    Returns:
    --------
    score : float
        ROC-AUC на валидационной выборке
    """
    # Проверяем, что model_name - это реальная модель с параметрами в config
    if model_name not in config or not isinstance(config[model_name], dict):
        raise ValueError(f"Модель '{model_name}' не найдена в конфигурации или её параметры не являются словарем")
    
    # Получение пространства параметров для текущей модели
    param_space = config[model_name]
    
    # Создание словаря параметров для текущего trial
    params = {}
    for param_name, param_values in param_space.items():
        if isinstance(param_values, list):
            params[param_name] = trial.suggest_categorical(param_name, param_values)
        elif isinstance(param_values, tuple):
            if isinstance(param_values[0], int):
                params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
            else:
                params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
    
    # Создание и обучение модели
    model = create_model(model_name, params)
    model.fit(X_train, y_train)
    
    # Предсказание на валидационной выборке
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Расчет ROC-AUC
    score = roc_auc_score(y_val, y_pred_proba)
    
    return score

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, config):
    """
    Оптимизация гиперпараметров модели с помощью Optuna
    
    Parameters:
    -----------
    model_name : str
        Название модели
    X_train, y_train : array-like
        Обучающие данные
    X_val, y_val : array-like
        Валидационные данные
    config : dict
        Конфигурация с параметрами моделей
        
    Returns:
    --------
    best_params : dict
        Лучшие параметры модели
    best_model : object
        Лучшая модель
    """
    # Проверяем, существует ли модель в конфигурации
    if model_name not in config:
        raise ValueError(f"Модель '{model_name}' не найдена в конфигурации")
    
    # Проверяем, что для модели определены параметры
    if not isinstance(config[model_name], dict):
        raise TypeError(f"Параметры для модели '{model_name}' должны быть словарем, получено: {type(config[model_name])}")
    
    # Устанавливаем verbosity для подавления вывода предупреждений
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, model_name, X_train, y_train, X_val, y_val, config),
        n_trials=config.get('n_trials', 20)  # Используем n_trials из config или значение по умолчанию
    )
    
    # Получение лучших параметров
    best_params = study.best_params
    
    # Создание и обучение лучшей модели
    best_model = create_model(model_name, best_params)
    best_model.fit(X_train, y_train)
    
    return best_params, best_model

def calculate_scores(model, X, df):
    """
    Расчет вероятностей дефолта и скоров
    
    Parameters:
    -----------
    model : object
        Обученная модель
    X : array-like
        Признаки
    df : pd.DataFrame
        Исходный DataFrame
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame с добавленными колонками pd и score
    """
    # Создаем копию DataFrame, чтобы не модифицировать исходный
    result_df = df.copy()
    
    # Расчет вероятностей дефолта
    proba = model.predict_proba(X)[:, 1]
    
    # Устанавливаем ограничения для предотвращения деления на 0 или логарифма от 0
    proba = np.clip(proba, 1e-15, 1-1e-15)
    
    # Расчет скоров через логиты
    score = np.log(proba / (1 - proba))
    
    # Добавление колонок в DataFrame
    result_df['pd'] = proba
    result_df['score'] = score
    
    return result_df

def plot_roc_curves(models, X_train, X_test, X_all, y_train, y_test, y_all):
    """
    Построение ROC-кривых для всех выборок
    
    Parameters:
    -----------
    models : dict
        Словарь с моделями
    X_train, X_test, X_all : array-like
        Признаки для разных выборок
    y_train, y_test, y_all : array-like
        Целевые переменные для разных выборок
    """
    plt.figure(figsize=(10, 8))
    
    # Цвета для разных моделей
    model_colors = {'catboost': 'purple', 'xgboost': 'blue', 'lightgbm': 'green'}
    
    # Стили линий для разных выборок
    sample_styles = {'train': ':', 'test': '-'}
    
    for name, model in models.items():
        model_color = model_colors.get(name, 'gray')
        
        for X, y, sample_name in [(X_train, y_train, 'train'), 
                                (X_test, y_test, 'test')]:
            y_pred_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            gini = 2 * roc_auc - 1
            
            plt.plot(fpr, tpr, 
                    label=f'{name}-{sample_name} (gini={gini:.2f})',
                    color=model_color,
                    linestyle=sample_styles[sample_name],
                    alpha=0.7)
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые для различных моделей и выборок')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show() 