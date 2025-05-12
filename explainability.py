"""
Модуль для анализа и интерпретации предсказаний модели
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from scipy.stats import ks_2samp

class ModelExplainer:
    def __init__(self, model, feature_names, categorical_features=None, random_state=42):
        """
        Инициализация объяснителя модели
        
        Parameters:
        -----------
        model : object
            Обученная модель
        feature_names : list
            Список названий признаков
        categorical_features : list, optional
            Список индексов категориальных признаков
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.scaler = StandardScaler()
        self.random_state = random_state
        
    def explain_prediction_lime(self, X, instance_idx, num_features=10):
        """
        Объяснение предсказания для конкретного наблюдения с помощью LIME
        
        Parameters:
        -----------
        X : array-like
            Матрица признаков
        instance_idx : int
            Индекс наблюдения для объяснения
        num_features : int
            Количество признаков для отображения
            
        Returns:
        --------
        exp : LimeExplanation
            Объект с объяснением
        """
        # Преобразуем X в numpy array, если это DataFrame
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Масштабируем данные с проверкой на некорректные значения
        X_scaled = np.zeros_like(X_array, dtype=np.float64)
        for i in range(X_array.shape[1]):
            # Заменяем inf и -inf на максимальные/минимальные значения
            col = X_array[:, i].copy()
            col[np.isinf(col)] = np.nan
            max_val = np.nanmax(col)
            min_val = np.nanmin(col)
            col[np.isnan(col)] = max_val if np.isinf(max_val) else min_val
            
            # Проверяем на нулевое стандартное отклонение
            std = np.std(col)
            if std < 1e-10:  # Если стандартное отклонение слишком маленькое
                X_scaled[:, i] = col - np.mean(col)  # Только центрируем
            else:
                X_scaled[:, i] = (col - np.mean(col)) / std
            
        # Проверяем на наличие NaN и inf
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("Предупреждение: Обнаружены NaN или inf значения после масштабирования")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Создаем объяснитель LIME с дополнительными параметрами
        explainer = lime_tabular.LimeTabularExplainer(
            X_scaled,
            feature_names=self.feature_names,
            class_names=['Хороший', 'Плохой'],
            categorical_features=self.categorical_features,
            mode='classification',
            discretize_continuous=True,
            kernel_width=3,  # Увеличиваем ширину ядра
            random_state=42,  # Фиксируем случайное состояние
            discretizer='quartile'  # Используем квартили для дискретизации
        )
        
        try:
            # Получаем объяснение с дополнительными параметрами
            exp = explainer.explain_instance(
                X_scaled[instance_idx],
                self.model.predict_proba,
                num_features=num_features,
                top_labels=1,  # Объясняем только для предсказанного класса
                num_samples=500  # Уменьшаем количество сэмплов
            )
            return exp
        except Exception as e:
            print(f"Ошибка при создании объяснения LIME: {str(e)}")
            print("Пробуем альтернативный подход...")
            
            # Альтернативный подход с еще меньшим количеством сэмплов
            exp = explainer.explain_instance(
                X_scaled[instance_idx],
                self.model.predict_proba,
                num_features=num_features,
                top_labels=1,
                num_samples=200  # Еще уменьшаем количество сэмплов
            )
            return exp
    
    def plot_lime_explanation(self, exp, instance_idx):
        """
        Визуализация объяснения LIME
        
        Parameters:
        -----------
        exp : LimeExplanation
            Объект с объяснением
        instance_idx : int
            Индекс наблюдения
        """
        try:
            if exp is None:
                print(f"Предупреждение: Не удалось создать объяснение для наблюдения {instance_idx}")
                return
                
            plt.figure(figsize=(10, 6))
            exp.as_pyplot_figure()
            plt.title(f'Объяснение LIME для наблюдения {instance_idx}')
            plt.tight_layout()
            
            # Проверяем, есть ли данные для отображения
            if len(plt.gcf().axes) == 0:
                print(f"Предупреждение: Нет данных для отображения в объяснении {instance_idx}")
                plt.close()
                return
                
            plt.show()
        except Exception as e:
            print(f"Ошибка при визуализации объяснения LIME для наблюдения {instance_idx}: {str(e)}")
            plt.close()  # Закрываем фигуру в случае ошибки
    
    def explain_global_shap(self, X, sample_size=100):
        """
        Глобальное объяснение модели с помощью SHAP
        
        Parameters:
        -----------
        X : array-like
            Матрица признаков
        sample_size : int
            Размер выборки для расчета SHAP значений
            
        Returns:
        --------
        shap_values : array
            SHAP значения
        """
        # Выбираем подвыборку для расчета SHAP значений
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Выбираем подвыборку для расчета SHAP значений
        if len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = X
            
        # Создаем объяснитель SHAP
        explainer = shap.TreeExplainer(self.model)
        
        # Рассчитываем SHAP значения
        shap_values = explainer.shap_values(X_sample)
        
        return shap_values, X_sample
    
    def plot_shap_summary(self, shap_values, X_sample):
        """
        Визуализация суммарного графика SHAP
        
        Parameters:
        -----------
        shap_values : array
            SHAP значения
        X_sample : array-like
            Матрица признаков для подвыборки
        """
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.title('Сводный график SHAP')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_dependence(self, shap_values, X_sample, feature_name):
        """
        Визуализация зависимости SHAP значений от признака
        
        Parameters:
        -----------
        shap_values : array
            SHAP значения
        X_sample : array-like
            Матрица признаков для подвыборки
        feature_name : str
            Название признака для анализа
        """
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            shap_values,
            X_sample,
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'График зависимости SHAP для признака {feature_name}')
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_importance(self, X, y, threshold=0.5):
        """
        Анализ важности признаков для разных групп заемщиков
        
        Parameters:
        -----------
        X : array-like
            Матрица признаков
        y : array-like
            Целевая переменная
        threshold : float
            Порог для разделения на группы
        """
        # Преобразуем X в DataFrame, если это еще не DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
            
        # Получаем предсказания модели
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Рассчитываем SHAP значения
        shap_values, X_sample = self.explain_global_shap(X)
        
        # Создаем DataFrame с результатами
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Создаем DataFrame с результатами
        results = pd.DataFrame({
            'Признак': self.feature_names,
            'Важность SHAP': mean_shap_values
        })
        
        # Добавляем предсказанный класс для каждого наблюдения
        results_with_predictions = []
        for i in range(len(X)):
            results_with_predictions.append({
                'Признак': self.feature_names,
                'Важность SHAP': mean_shap_values,
                'Предсказанный класс': y_pred[i]
            })
        
        results = pd.concat([pd.DataFrame(d) for d in results_with_predictions], ignore_index=True)
        
        # Группируем по предсказанному классу
        grouped = results.groupby('Предсказанный класс')['Важность SHAP'].mean()
        
        # Визуализация
        plt.figure(figsize=(12, 6))
        grouped.plot(kind='bar')
        plt.title('Средняя важность признаков по предсказанному классу')
        plt.xlabel('Предсказанный класс')
        plt.ylabel('Среднее |SHAP значение|')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        return results
    def create_analysis_report(self, X, y, threshold, model_name, shap_values, X_sample, results):
        """
        Создание и сохранение отчета в Excel
        
        Parameters:
        -----------
        X : array-like
            Матрица признаков
        y : array-like
            Целевая переменная
        threshold : float
            Порог классификации
        model_name : str
            Название модели
        shap_values : array
            SHAP значения
        X_sample : array-like
            Выборка для SHAP анализа
        results : pd.DataFrame
            Результаты анализа важности признаков
        """
        from datetime import datetime
        import os
        
        # Создаем директорию для отчетов, если её нет
        os.makedirs('reports', exist_ok=True)
        
        # Генерируем имя файла с датой и временем
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'reports/model_analysis_{model_name}_{timestamp}.xlsx'
        
        # Получаем предсказания модели
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Рассчитываем метрики
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        balanced_accuracy = (tp/(tp+fn) + tn/(tn+fp)) / 2
        f1 = f1_score(y, y_pred, average='weighted')
        ks_stat, _ = ks_2samp(y_pred_proba[y == 0], y_pred_proba[y == 1])
        auc = roc_auc_score(y, y_pred_proba)
        
        # Создаем Excel writer
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 1. Общая информация
            general_info = pd.DataFrame({
                'Параметр': [
                    'Название модели',
                    'Дата анализа',
                    'Количество наблюдений',
                    'Количество признаков',
                    'Порог классификации',
                    'Доля положительного класса'
                ],
                'Значение': [
                    model_name,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(X),
                    len(self.feature_names),
                    f'{threshold:.3f}',
                    f'{y.mean():.3f}'
                ]
            })
            general_info.to_excel(writer, sheet_name='Общая информация', index=False)
            
            # 2. Метрики качества
            metrics_info = pd.DataFrame({
                'Метрика': [
                    'Balanced Accuracy',
                    'Balanced F1 Score',
                    'KS-статистика',
                    'AUC',
                    'Accuracy',
                    'Precision',
                    'Recall',
                    'Specificity',
                    'False Positive Rate',
                    'False Negative Rate'
                ],
                'Значение': [
                    f'{balanced_accuracy:.3f}',
                    f'{f1:.3f}',
                    f'{ks_stat:.3f}',
                    f'{auc:.3f}',
                    f'{(tp + tn)/(tp + tn + fp + fn):.3f}',
                    f'{tp/(tp + fp):.3f}',
                    f'{tp/(tp + fn):.3f}',
                    f'{tn/(tn + fp):.3f}',
                    f'{fp/(fp + tn):.3f}',
                    f'{fn/(fn + tp):.3f}'
                ]
            })
            metrics_info.to_excel(writer, sheet_name='Метрики качества', index=False)
            
            # 3. Матрица ошибок
            confusion_matrix_df = pd.DataFrame({
                'Предсказанный класс': ['Отрицательный', 'Положительный'],
                'Фактический класс Отрицательный': [tn, fp],
                'Фактический класс Положительный': [fn, tp]
            })
            confusion_matrix_df.to_excel(writer, sheet_name='Матрица ошибок', index=False)
            
            # 4. Важность признаков
            feature_importance = pd.DataFrame({
                'Признак': self.feature_names,
                'Среднее |SHAP значение|': np.abs(shap_values).mean(axis=0),
                'Максимальное |SHAP значение|': np.abs(shap_values).max(axis=0),
                'Стандартное отклонение SHAP': np.std(shap_values, axis=0)
            }).sort_values('Среднее |SHAP значение|', ascending=False)
            feature_importance.to_excel(writer, sheet_name='Важность признаков', index=False)
            
            # 5. Взаимодействия признаков
            interactions = []
            for i, feat1 in enumerate(self.feature_names):
                for j, feat2 in enumerate(self.feature_names):
                    if i != j:
                        corr = np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]
                        interactions.append({
                            'Признак 1': feat1,
                            'Признак 2': feat2,
                            'Корреляция SHAP': corr
                        })
            interactions_df = pd.DataFrame(interactions)
            interactions_df = interactions_df[interactions_df['Корреляция SHAP'].abs() > 0.3].sort_values('Корреляция SHAP', ascending=False)
            interactions_df.to_excel(writer, sheet_name='Взаимодействия признаков', index=False)
            
            # 6. Статистика по классам
            class_stats = pd.DataFrame({
                'Класс': ['Хороший', 'Плохой'],
                'Количество': [sum(y == 0), sum(y == 1)],
                'Доля': [sum(y == 0)/len(y), sum(y == 1)/len(y)]
            })
            class_stats.to_excel(writer, sheet_name='Статистика классов', index=False)
            
            # 7. Топ-10 важных признаков с их статистикой
            top_features = feature_importance.head(10)['Признак'].tolist()
            top_features_stats = X[top_features].describe().T
            top_features_stats.to_excel(writer, sheet_name='Статистика топ-признаков')
            
            # 8. Результаты анализа важности по классам
            results.to_excel(writer, sheet_name='Важность по классам', index=False)
        
        print(f"\nОтчет сохранен в файл: {filename}")
        return filename


def plot_feature_interactions(shap_values, X_sample, top_n=5):
    """
    Анализ взаимодействий между признаками
    
    Parameters:
    -----------
    shap_values : array
        SHAP значения
    X_sample : array-like
        Матрица признаков для подвыборки
    top_n : int
        Количество топовых признаков для анализа
    """
    # Находим топ-N важных признаков
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-top_n:]
    
    # Получаем названия топовых признаков
    if isinstance(X_sample, pd.DataFrame):
        top_feature_names = X_sample.columns[top_features_idx].tolist()
    else:
        top_feature_names = [f'Признак {i+1}' for i in range(top_n)]
    
    # Создаем матрицу взаимодействий
    interaction_matrix = np.zeros((top_n, top_n))
    
    for i, feat1 in enumerate(top_features_idx):
        for j, feat2 in enumerate(top_features_idx):
            if i != j:
                # Рассчитываем корреляцию между SHAP значениями
                interaction_matrix[i, j] = np.corrcoef(
                    shap_values[:, feat1],
                    shap_values[:, feat2]
                )[0, 1]
    
    # Визуализация
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        interaction_matrix,
        annot=True,
        cmap='RdBu',
        center=0,
        xticklabels=top_feature_names,
        yticklabels=top_feature_names
    )
    plt.title('Матрица взаимодействий признаков')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show() 