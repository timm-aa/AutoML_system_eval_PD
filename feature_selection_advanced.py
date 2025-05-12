"""
Модуль для продвинутого отбора признаков
"""

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict


def find_sharp_drop_point(df: pd.DataFrame, x_col: str, y_col: str, threshold: float = 0.05) -> int:
    """
    Находит точку резкого спада на кривой метрики качества
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными
    x_col : str
        Название колонки с x-координатами (количество признаков)
    y_col : str
        Название колонки с y-координатами (значение метрики)
    threshold : float
        Порог относительного изменения метрики (по умолчанию 5%)
        
    Returns:
    --------
    optimal_point : int
        Оптимальное значение x в точке резкого спада
    """
    # Сортируем данные по количеству признаков (по убыванию)
    df = df.sort_values(by=x_col, ascending=False).reset_index(drop=True)
    
    # Вычисляем производную (разницу между соседними значениями)
    df['derivative'] = df[y_col].diff()
    
    # Вычисляем относительное изменение метрики
    df['relative_change'] = df[y_col].pct_change()
    
    # Находим точку с максимальной производной
    max_derivative_idx = df['derivative'].idxmax()
    min_derivative_idx = df['derivative'].idxmin()
    # Находим первую точку, где относительное изменение превышает порог
    threshold_idx = df[df['relative_change'] > threshold].index[0] if any(df['relative_change'] > threshold) else None

    # Выбираем более раннюю точку из двух методов
    if threshold_idx is not None and threshold_idx > max_derivative_idx:
        if threshold_idx > min_derivative_idx:
            optimal_idx = threshold_idx
        else:
            optimal_idx = min_derivative_idx - 1
    else:
        if max_derivative_idx > min_derivative_idx:
            optimal_idx = max_derivative_idx
        else:
            optimal_idx = min_derivative_idx - 1
    
    # Получаем оптимальное количество признаков
    optimal_point = int(df.iloc[optimal_idx][x_col])
    
    # Выводим информацию о выбранной точке
    print(f"\nАнализ точки резкого спада:")
    print(f"Максимальное изменение метрики: {df.iloc[max_derivative_idx]['derivative']:.4f}")
    if threshold_idx is not None:
        print(f"Относительное изменение в точке порога: {df.iloc[threshold_idx]['relative_change']:.2%}")
    print(f"Выбрано количество признаков: {optimal_point}")
    
    return optimal_point

def feature_importance_selection(X: pd.DataFrame, y: pd.Series, config: dict, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Tuple[List[str], pd.DataFrame, int]:
    """
    Отбор признаков на основе важности признаков с использованием метода резкого спада
    """
    n_features = X.shape[1]
    results = []
    current_features = X.columns.tolist()
    use_test_data = X_test is not None and y_test is not None
    
    print("\n==== Отбор по Feature Importance ====")
    if not use_test_data:
        raise ValueError("Необходимо предоставить тестовые данные (X_test, y_test) для отбора признаков")

    # Базовые параметры модели
    model_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': config['random_state']
    }

    while len(current_features) > config['min_features_fi']:
        # Определяем количество признаков для удаления
        if n_features >= 300:
            n_to_remove = int(len(current_features) * 0.2)
        elif n_features >= 100:
            n_to_remove = int(len(current_features) * 0.1)
        elif n_features >= 50:
            n_to_remove = int(len(current_features) * 0.05)
        else:
            n_to_remove = 1
            
        # Обучаем модель
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X[current_features], y)

        # Оцениваем качество
        y_pred = model.predict_proba(X[current_features])[:, 1]
        train_score = 2*roc_auc_score(y, y_pred) - 1
        
        y_test_pred = model.predict_proba(X_test[current_features])[:, 1]
        test_score = 2*roc_auc_score(y_test, y_test_pred) - 1
        print(f"Gini с {len(current_features)} признаками: Train: {train_score:.4f}, Test: {test_score:.4f}")

        # Получаем важность признаков
        importance = pd.Series(model.feature_importances_, index=current_features)
        importance = importance.sort_values(ascending=False)
        
        features_to_remove = importance.index[-n_to_remove:].tolist()
        
        # Удаляем наименее важные признаки
        current_features = importance.index[:-n_to_remove].tolist()
        n_features = len(current_features)
        print(f"Удалено {n_to_remove} признаков с наименьшей важностью: {', '.join(features_to_remove)}")
        print(f"Осталось признаков: {len(current_features)}")
        
        # Сохраняем результаты
        results.append({
            'n_features': n_features,
            'train_roc_auc': train_score,
            'test_roc_auc': test_score
        })
    
    # Строим график
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 8))
    plt.style.use('bmh')
    
    # Основные линии
    plt.plot(results_df['n_features'], results_df['train_roc_auc'], 
             label='Обучающая выборка', linewidth=2, color='#2ecc71', alpha=0.8)
    plt.plot(results_df['n_features'], results_df['test_roc_auc'], 
             label='Тестовая выборка', linewidth=2, color='#3498db', alpha=0.8)
    
    # Добавляем точки на линии
    plt.scatter(results_df['n_features'], results_df['train_roc_auc'], 
               color='#2ecc71', s=50, alpha=0.6)
    plt.scatter(results_df['n_features'], results_df['test_roc_auc'], 
               color='#3498db', s=50, alpha=0.6)
    
    # Находим точку резкого спада
    
    
    optimal_n_features = find_sharp_drop_point(results_df, 'n_features', 'test_roc_auc')
    optimal_score = results_df[results_df['n_features'] == optimal_n_features]['test_roc_auc'].iloc[0]
    
    # Отмечаем точку резкого спада
    plt.scatter(optimal_n_features, optimal_score, 
               color='#e74c3c', s=150, label='Точка резкого спада', 
               zorder=5, edgecolor='black', linewidth=2)
    
    # Добавляем аннотацию
    plt.annotate(f'Оптимальное количество признаков: {optimal_n_features}', 
                xy=(optimal_n_features, optimal_score),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Оформление графика
    plt.title('Зависимость Gini от количества признаков (Feature Importance)', 
             fontsize=14, pad=20)
    plt.xlabel('Количество признаков', fontsize=12)
    plt.ylabel('Gini', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    plt.show()
    
    return current_features, results_df, optimal_n_features

def permutation_importance_selection(X: pd.DataFrame, y: pd.Series, config: dict, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> Tuple[List[str], pd.DataFrame, int]:
    """
    Отбор признаков на основе permutation importance с использованием метода резкого спада
    """
    current_features = X.columns.tolist()
    results = []
    use_test_data = X_test is not None and y_test is not None
    
    print("\n==== Отбор по Permutation Importance ====")
    if not use_test_data:
        raise ValueError("Необходимо предоставить тестовые данные (X_test, y_test) для отбора признаков")
    
    # Базовые параметры модели
    model_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': config['random_state']
    }

    while len(current_features) > config['min_features_pi']:
        # Обучаем модель
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X[current_features], y)

        # Получаем permutation importance
        perm_importance = permutation_importance(
            model, X[current_features], y,
            n_repeats=10,
            random_state=config['random_state']
        )

        # Находим наименее важный признак
        importance = pd.Series(perm_importance.importances_mean, index=current_features)
        least_important = importance.idxmin()
        
        # Оцениваем качество
        y_pred = model.predict_proba(X[current_features])[:, 1]
        train_score = 2*roc_auc_score(y, y_pred) - 1
        
        y_test_pred = model.predict_proba(X_test[current_features])[:, 1]
        test_score = 2*roc_auc_score(y_test, y_test_pred) - 1
        print(f"Gini с {len(current_features)} признаками: Train: {train_score:.4f}, Test: {test_score:.4f}")
        
        # Удаляем наименее важный признак
        current_features.remove(least_important)
        print(f"Удален признак '{least_important}' с наименьшей важностью")
        print(f"Осталось признаков: {len(current_features)}")
        
        # Сохраняем результаты
        results.append({
            'n_features': len(current_features),
            'train_roc_auc': train_score,
            'test_roc_auc': test_score,
            'removed_feature': least_important
        })
    
    # Строим график
    results_df = pd.DataFrame(results)
    plt.figure(figsize=(12, 8))
    plt.style.use('bmh')
    
    # Основные линии
    plt.plot(results_df['n_features'], results_df['train_roc_auc'], 
             label='Обучающая выборка', linewidth=2, color='#2ecc71', alpha=0.8)
    plt.plot(results_df['n_features'], results_df['test_roc_auc'], 
             label='Тестовая выборка', linewidth=2, color='#3498db', alpha=0.8)
    
    # Добавляем точки на линии
    plt.scatter(results_df['n_features'], results_df['train_roc_auc'], 
               color='#2ecc71', s=50, alpha=0.6)
    plt.scatter(results_df['n_features'], results_df['test_roc_auc'], 
               color='#3498db', s=50, alpha=0.6)
    
    # Находим точку резкого спада
    optimal_n_features = find_sharp_drop_point(results_df, 'n_features', 'test_roc_auc')
    optimal_score = results_df[results_df['n_features'] == optimal_n_features]['test_roc_auc'].iloc[0]
    
    # Отмечаем точку резкого спада
    plt.scatter(optimal_n_features, optimal_score, 
               color='#e74c3c', s=150, label='Точка резкого спада', 
               zorder=5, edgecolor='black', linewidth=2)
    
    # Добавляем аннотацию
    plt.annotate(f'Оптимальное количество признаков: {optimal_n_features}', 
                xy=(optimal_n_features, optimal_score),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Оформление графика
    plt.title('Зависимость Gini от количества признаков (Permutation Importance)', 
             fontsize=14, pad=20)
    plt.xlabel('Количество признаков', fontsize=12)
    plt.ylabel('Gini', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    
    plt.show()
    
    return current_features, results_df, optimal_n_features

def remove_correlated_features(X: pd.DataFrame, features: List[str], config: dict) -> List[str]:
    """
    Удаление коррелированных признаков
    
    Parameters:
    -----------
    X : pd.DataFrame
        Признаки
    features : List[str]
        Список признаков для анализа
    config : dict
        Словарь с параметрами отбора признаков
        
    Returns:
    --------
    selected_features : List[str]
        Отобранные признаки
    """
    # Строим корреляционную матрицу
    corr_matrix = X[features].corr().abs()
    
    # Находим пары признаков с корреляцией выше порога
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(round(upper[column], 2) >= config['correlation_threshold'])]
    
    # Удаляем один из коррелированных признаков
    selected_features = [col for col in features if col not in to_drop]
    
    # Строим итоговую корреляционную матрицу
    plt.figure(figsize=(12, 8))
    sns.heatmap(X[features].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица признаков')
    plt.show()
    
    return selected_features

def select_features(X: pd.DataFrame, y: pd.Series, config: dict, X_test: pd.DataFrame = None, y_test: pd.Series = None) -> List[str]:
    """
    Основная функция отбора признаков
    
    Parameters:
    -----------
    X : pd.DataFrame
        Признаки для обучения
    y : pd.Series
        Целевая переменная для обучения
    config : dict
        Словарь с параметрами отбора признаков
    X_test : pd.DataFrame, optional
        Признаки тестового набора для проверки
    y_test : pd.Series, optional
        Целевая переменная тестового набора для проверки
        
    Returns:
    --------
    selected_features : List[str]
        Отобранные признаки
    """
    # Проверка наличия тестовых данных
    if X_test is None or y_test is None:
        raise ValueError("Необходимо предоставить тестовые данные (X_test, y_test) для отбора признаков")
    
    print("Отбор признаков будет проводиться с использованием тестовой выборки для оценки качества")
    
    # Проверяем, что все признаки в X есть в X_test
    missing_cols = [col for col in X.columns if col not in X_test.columns]
    if missing_cols:
        raise ValueError(f"В тестовых данных отсутствуют следующие признаки: {missing_cols}")
    
    # Базовые параметры модели для оценки на всех признаках
    model_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': config['random_state']
    }
    
    # Оцениваем производительность на полном наборе признаков
    print("\n--- Оценка базовой модели на всех признаках ---")
    base_model = lgb.LGBMClassifier(**model_params)
    base_model.fit(X, y)
    
    base_train_score = 2*roc_auc_score(y, base_model.predict_proba(X)[:, 1]) - 1
    base_test_score = 2*roc_auc_score(y_test, base_model.predict_proba(X_test)[:, 1]) - 1
    print(f"Базовая модель (все признаки): Gini на обучении: {base_train_score:.4f}, Gini на тесте: {base_test_score:.4f}")
    
    # Отбор на основе feature importance
    fi_features, fi_results, fi_optimal_n = feature_importance_selection(X, y, config, X_test, y_test)
    print(f"\nОптимальное количество признаков по Feature Importance: {fi_optimal_n}")
    
    # Получаем топ-N признаков по feature importance
    fi_model = lgb.LGBMClassifier(**model_params)
    fi_model.fit(X, y)
    fi_importance = pd.Series(fi_model.feature_importances_, index=X.columns)
    fi_importance = fi_importance.sort_values(ascending=False)
    fi_top_features = fi_importance.index[:fi_optimal_n].tolist()
    print(f"Отобрано {len(fi_top_features)} признаков методом feature importance")
    
    # Отбор на основе permutation importance только для признаков, отобранных feature importance
    pi_features, pi_results, pi_optimal_n = permutation_importance_selection(
        X[fi_top_features], y, config, X_test[fi_top_features], y_test
    )
    print(f"\nОптимальное количество признаков по Permutation Importance: {pi_optimal_n}")
    
    # Получаем топ-N признаков по permutation importance
    pi_model = lgb.LGBMClassifier(**model_params)
    pi_model.fit(X[fi_top_features], y)
    perm_importance = permutation_importance(
        pi_model, X[fi_top_features], y,
        n_repeats=10,
        random_state=config['random_state']
    )
    pi_importance = pd.Series(perm_importance.importances_mean, index=fi_top_features)
    pi_importance = pi_importance.sort_values(ascending=False)
    pi_top_features = pi_importance.index[:pi_optimal_n].tolist()
    print(f"Отобрано {len(pi_top_features)} признаков методом permutation importance")
    
    # Удаляем коррелированные признаки из pi_top_features (финальных признаков)
    final_features = remove_correlated_features(X, pi_top_features, config)
    print(f"Итоговое количество признаков после удаления корреляций: {len(final_features)}")
    
    # Выведем итоговую производительность
    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(X[final_features], y)
    
    train_score = 2*roc_auc_score(y, final_model.predict_proba(X[final_features])[:, 1]) - 1
    test_score = 2*roc_auc_score(y_test, final_model.predict_proba(X_test[final_features])[:, 1]) - 1
    
    print(f"\nИтоговая производительность модели на отобранных признаках:")
    print(f"Gini на обучающих данных: {train_score:.4f}")
    print(f"Gini на тестовых данных: {test_score:.4f}")
    
    # Сравнение с моделью на всех признаках
    print(f"\nСравнение с моделью на всех признаках:")
    print(f"Gini на обучающих данных (все признаки): {base_train_score:.4f}")
    print(f"Gini на тестовых данных (все признаки): {base_test_score:.4f}")
    
    # Изменение в процентах
    train_change = (train_score - base_train_score) / base_train_score * 100
    test_change = (test_score - base_test_score) / base_test_score * 100
    
    print(f"Изменение на обучающих данных: {train_change:.2f}%")
    print(f"Изменение на тестовых данных: {test_change:.2f}%")
    
    print(f"\nИтоговое уменьшение размерности: с {len(X.columns)} до {len(final_features)} признаков ({len(final_features)/len(X.columns)*100:.1f}%)")
    print(f"Уменьшение размерности на {100 - len(final_features)/len(X.columns)*100:.1f}%")
    
    return final_features 