"""
Модуль для оптимизации порога разделения дефолтов и не дефолтов
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import ks_2samp

def find_optimal_threshold(y_true, y_pred_proba, method='roc', 
                          target_recall=None, cost_fp=None, cost_fn=None):
    """
    Поиск оптимального порога разделения
    
    Parameters:
    -----------
    y_true : array-like
        Истинные метки классов
    y_pred_proba : array-like
        Предсказанные вероятности
    method : str
        Метод поиска порога:
        - 'business': на основе бизнес-требований
        - 'roc': точка на ROC-кривой
        - 'f1': максимизация F1-score
        - 'hybrid': комбинированный подход
    target_recall : float, optional
        Целевой уровень recall для дефолтных клиентов
    cost_fp : float, optional
        Стоимость ошибки первого рода (отказ хорошему заемщику)
    cost_fn : float, optional
        Стоимость ошибки второго рода (выдача плохому заемщику)
        
    Returns:
    --------
    optimal_threshold : float
        Оптимальный порог
    metrics : dict
        Метрики при оптимальном пороге
    """
    if method == 'business':
        return _find_business_threshold(y_true, y_pred_proba, target_recall, cost_fp, cost_fn)
    elif method == 'roc':
        return _find_roc_threshold(y_true, y_pred_proba)
    elif method == 'f1':
        return _find_f1_threshold(y_true, y_pred_proba)
    elif method == 'hybrid':
        return _find_hybrid_threshold(y_true, y_pred_proba, target_recall)
    else:
        raise ValueError(f"Unknown method: {method}")

def _find_business_threshold(y_true, y_pred_proba, target_recall, cost_fp, cost_fn):
    """Поиск порога на основе бизнес-требований"""
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_threshold = 0.5
    min_cost = float('inf')
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Расчет общей стоимости ошибок
        total_cost = fp * cost_fp + fn * cost_fn
        
        if total_cost < min_cost:
            min_cost = total_cost
            best_threshold = threshold
            
    return best_threshold, {
        'threshold': best_threshold,
        'total_cost': min_cost,
        'method': 'business'
    }

def _find_roc_threshold(y_true, y_pred_proba):
    """Поиск порога на основе ROC-кривой"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, {
        'threshold': optimal_threshold,
        'tpr': tpr[optimal_idx],
        'fpr': fpr[optimal_idx],
        'method': 'roc'
    }

def _find_f1_threshold(y_true, y_pred_proba):
    """Поиск порога на основе F1-score"""
    thresholds = np.arange(0.1, 1.0, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            
    return best_threshold, {
        'threshold': best_threshold,
        'f1_score': best_f1,
        'method': 'f1'
    }

def _find_hybrid_threshold(y_true, y_pred_proba, target_recall):
    """Комбинированный подход к поиску порога"""
    # Получаем пороги разными методами
    business_threshold, business_metrics = _find_business_threshold(
        y_true, y_pred_proba, target_recall, cost_fp=1, cost_fn=5)
    roc_threshold, roc_metrics = _find_roc_threshold(y_true, y_pred_proba)
    f1_threshold, f1_metrics = _find_f1_threshold(y_true, y_pred_proba)
    
    # Взвешенное среднее порогов
    weights = {'business': 0.4, 'roc': 0.3, 'f1': 0.3}
    hybrid_threshold = (
        business_threshold * weights['business'] +
        roc_threshold * weights['roc'] +
        f1_threshold * weights['f1']
    )
    
    return hybrid_threshold, {
        'threshold': hybrid_threshold,
        'business_threshold': business_threshold,
        'roc_threshold': roc_threshold,
        'f1_threshold': f1_threshold,
        'method': 'hybrid'
    }

def plot_threshold_analysis(y_true, y_pred_proba, optimal_threshold):
    """
    Визуализация анализа порогов
    
    Parameters:
    -----------
    y_true : array-like
        Истинные метки классов
    y_pred_proba : array-like
        Предсказанные вероятности
    optimal_threshold : float
        Оптимальный порог
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('Доля ложных срабатываний (FPR)')
    ax1.set_ylabel('Доля истинных срабатываний (TPR)')
    ax1.set_title('ROC-кривая')
    
    # Распределение вероятностей
    df_plot = pd.DataFrame({
        'Вероятность': y_pred_proba,
        'Класс': y_true
    })
    
    # Создаем график
    sns.histplot(data=df_plot, x='Вероятность', hue='Класс', bins=50, ax=ax2)
    
    
    ax2.axvline(x=optimal_threshold, color='r', linestyle='--', 
                label=f'Оптимальный порог: {optimal_threshold:.2f}')
    ax2.set_title('Распределение вероятностей')
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    # Расчет метрик
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Balanced Accuracy
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    balanced_accuracy = (tp/(tp+fn) + tn/(tn+fp)) / 2
    
    # Balanced F1 Score
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Kolmogorov-Smirnov
    ks_stat, _ = ks_2samp(y_pred_proba[y_true == 0], y_pred_proba[y_true == 1])
    
    # AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Создаем таблицу с метриками
    metrics_df = pd.DataFrame({
        'Метрика': ['Balanced Accuracy', 'F1 Score', 'KS-статистика', 'AUC'],
        'Значение': [f'{balanced_accuracy:.3f}', f'{f1:.3f}', f'{ks_stat:.3f}', f'{auc:.3f}']
    })
    
    # Выводим таблицу
    print("\nМетрики качества модели:")
    print(metrics_df.to_string(index=False))

def analyze_threshold_impact(df, threshold, target_col='target', score_col='score'):
    """
    Анализ влияния порога на бизнес-метрики
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame с данными
    threshold : float
        Порог разделения
    target_col : str
        Название колонки с целевой переменной
    score_col : str
        Название колонки со скорами
    """
    # Применяем порог
    df['predicted'] = (df[score_col] >= threshold).astype(int)
    
    # Считаем метрики
    total = len(df)
    approved = (df['predicted'] == 0).sum()
    rejected = (df['predicted'] == 1).sum()
    
    # Среди одобренных
    approved_good = ((df['predicted'] == 0) & (df[target_col] == 0)).sum()
    approved_bad = ((df['predicted'] == 0) & (df[target_col] == 1)).sum()
    
    # Среди отклоненных
    rejected_good = ((df['predicted'] == 1) & (df[target_col] == 0)).sum()
    rejected_bad = ((df['predicted'] == 1) & (df[target_col] == 1)).sum()
    
    # Выводим результаты
    print(f"Всего заявок: {total}")
    print(f"Одобрено: {approved} ({approved/total:.1%})")
    print(f"Отклонено: {rejected} ({rejected/total:.1%})")
    print("\nСреди одобренных:")
    print(f"Хорошие заемщики: {approved_good} ({approved_good/approved:.1%})")
    print(f"Плохие заемщики: {approved_bad} ({approved_bad/approved:.1%})")
    print("\nСреди отклоненных:")
    print(f"Хорошие заемщики: {rejected_good} ({rejected_good/rejected:.1%})")
    print(f"Плохие заемщики: {rejected_bad} ({rejected_bad/rejected:.1%})") 