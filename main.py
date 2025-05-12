"""
Основной модуль для запуска полного пайплайна анализа данных
"""

import pandas as pd
import numpy as np
from config import PREPROCESSING_CONFIG, FEATURE_SELECTION_CONFIG, MODEL_CONFIG, THRESHOLD_CONFIG, EXPLAINABILITY_CONFIG
from preprocessing import prepare_data, create_preprocessing_pipeline
from feature_selection_advanced import select_features
from hyperparameter_tuning import optimize_hyperparameters, plot_roc_curves
from threshold_optimization import find_optimal_threshold, plot_threshold_analysis, analyze_threshold_impact
from explainability import ModelExplainer, plot_feature_interactions
import joblib
import os
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class CreditScoringPipeline:
    def __init__(self, config):
        """
        Инициализация пайплайна
        
        Parameters:
        -----------
        config : dict
            Конфигурация с параметрами:
            - random_state: int, seed для воспроизводимости
            - features: list, список всех признаков для обработки
            - target_column: str, название целевой переменной
            - feature_selection: dict, параметры отбора признаков
            - model_config: dict, параметры моделей
            - threshold_config: dict, параметры оптимизации порога
            - explainability_config: dict, параметры интерпретации
        """
        self.config = config
        self.preprocessor = None
        self.selected_features = None
        self.preproc_features = None
        self.best_models = {}
        self.optimal_thresholds = {}
        self.explainer = None
        
    def preprocess_data(self, df):
        """
        Предобработка данных
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датасет
            
        Returns:
        --------
        df : pd.DataFrame
            Подготовленный датасет
        """
        print("Предобработка данных...")
        df, self.preproc_features = prepare_data(
            df,
            self.config['target_column'],
            self.config
        )
        print("Предобработка завершена.")
        return df
    
    def select_best_features(self, df):
        """
        Отбор лучших признаков
        
        Parameters:
        -----------
        df : pd.DataFrame
            Подготовленный датасет
            
        Returns:
        --------
        selected_features : list
            Список отобранных признаков
        """
        print("Начало отбора признаков...")
        train_data = df[df['group'] == 'train']
        test_data = df[df['group'] == 'test']
        
        # Подготовка данных для отбора признаков
        X_train = train_data[self.preproc_features]
        y_train = train_data[self.config['target_column']]
        
        # Подготовка тестовых данных, если они есть
        X_test = None
        y_test = None
        if not test_data.empty:
            X_test = test_data[self.preproc_features]
            y_test = test_data[self.config['target_column']]
        
        # Отбор признаков с учетом тестовых данных
        self.selected_features = select_features(
            X_train,
            y_train,
            self.config['feature_selection'],
            X_test,
            y_test
        )
        print(f"Отбор признаков завершен. Отобрано {len(self.selected_features)} фичей.")
        return self.selected_features

    def optimize_models(self, df):
        """
        Оптимизация гиперпараметров моделей
        
        Parameters:
        -----------
        df : pd.DataFrame
            Подготовленный датасет
            
        Returns:
        --------
        best_models : dict
            Словарь с лучшими моделями
        """
        print("Начало оптимизации гиперпараметров моделей...")
        train_data = df[df['group'] == 'train']
        test_data = df[df['group'] == 'test']
        
        # Подготовка данных
        X_train = train_data[self.selected_features]
        y_train = train_data[self.config['target_column']]
        X_test = test_data[self.selected_features]
        y_test = test_data[self.config['target_column']]
        
        # Список доступных моделей
        available_models = ['catboost', 'xgboost', 'lightgbm']
        
        # Оптимизация каждой модели
        for model_name in available_models:
            # Проверяем, определена ли модель в конфигурации
            if model_name in self.config['model_config'] and isinstance(self.config['model_config'][model_name], dict):
                print(f"Оптимизация гиперпараметров для {model_name}...")
                try:
                    best_params, best_model = optimize_hyperparameters(
                        model_name,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        self.config['model_config']
                    )
                    self.best_models[model_name] = best_model
                    print(f"{model_name} optimization completed.")
                except Exception as e:
                    print(f"Ошибка при оптимизации {model_name}: {str(e)}")
            else:
                print(f"Пропуск модели {model_name}: не определена в конфигурации или параметры некорректны")
        
        if not self.best_models:
            raise ValueError("Не удалось оптимизировать ни одну модель. Проверьте конфигурацию.")
        
        # Построение ROC-кривых
        plot_roc_curves(
            self.best_models,
            X_train,
            X_test,
            pd.concat([X_train, X_test]),
            y_train,
            y_test,
            pd.concat([y_train, y_test])
        )
        
        return self.best_models
    
    def optimize_thresholds(self, df):
        """
        Оптимизация порогов для моделей
        
        Parameters:
        -----------
        df : pd.DataFrame
            Подготовленный датасет
            
        Returns:
        --------
        optimal_thresholds : dict
            Словарь с оптимальными порогами
        """
        print("Начало подбора порога классификации...")
        test_data = df[df['group'] == 'test']
        X_test = test_data[self.selected_features]
        y_test = test_data[self.config['target_column']]
        
        for model_name, model in self.best_models.items():
            print(f"Оптимизация порога для {model_name}...")
            # Получаем предсказания
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Находим оптимальный порог
            threshold, metrics = find_optimal_threshold(
                y_test,
                y_pred_proba,
                method=self.config['threshold_config']['method'],
                target_recall=self.config['threshold_config'].get('target_recall'),
                cost_fp=self.config['threshold_config'].get('cost_fp'),
                cost_fn=self.config['threshold_config'].get('cost_fn')
            )
            
            self.optimal_thresholds[model_name] = threshold
            
            # Визуализация
            plot_threshold_analysis(y_test, y_pred_proba, threshold)
            
            # Анализ влияния порога
            results_df = pd.DataFrame({
                'target': y_test,
                'score': y_pred_proba
            })
            analyze_threshold_impact(results_df, threshold)
            
        return self.optimal_thresholds
    
    def analyze_model(self, df, model_name):
        """
        Анализ и интерпретация модели
        
        Parameters:
        -----------
        df : pd.DataFrame
            Подготовленный датасет
        model_name : str
            Название модели для анализа
        """
        print(f"Начало анализа модели для {model_name}...")
        test_data = df[df['group'] == 'test']
        X_test = test_data[self.selected_features]
        y_test = test_data[self.config['target_column']]
        
        # Создаем объяснитель
        self.explainer = ModelExplainer(
            self.best_models[model_name],
            self.selected_features
        )
        
        # Глобальный анализ
        shap_values, X_sample = self.explainer.explain_global_shap(
            X_test,
            sample_size=self.config['explainability_config']['sample_size']
        )
        
        # Визуализация
        self.explainer.plot_shap_summary(shap_values, X_sample)
        
        # Анализ важности признаков
        results = self.explainer.analyze_feature_importance(
            X_test,
            y_test,
            threshold=self.optimal_thresholds[model_name]
        )
        
        # Анализ взаимодействий
        plot_feature_interactions(
            shap_values,
            X_sample,
            top_n=self.config['explainability_config']['top_features']
        )
        
        # Локальный анализ для нескольких примеров
        for idx in range(min(3, len(X_test))):
            exp = self.explainer.explain_prediction_lime(
                X_test,
                idx,
                num_features=self.config['explainability_config']['num_features']
            )
            self.explainer.plot_lime_explanation(exp, idx)
        
        # Создание и сохранение отчета
        report_path = self.explainer.create_analysis_report(
            X_test,
            y_test,
            self.optimal_thresholds[model_name],
            model_name,
            shap_values,
            X_sample,
            results
        )
        
        print(f"\nАнализ модели {model_name} завершен.")
        print(f"Полный отчет сохранен в файл: {report_path}")
        
        return report_path
    
    def save_best_model(self, model_name, model, threshold):
        """
        Сохранение лучшей модели и её порога в файл
        
        Parameters:
        -----------
        model_name : str
            Название модели
        model : object
            Объект модели
        threshold : float
            Оптимальный порог классификации
        """
        # Создаем директорию для моделей, если её нет
        os.makedirs('models', exist_ok=True)
        
        # Сохраняем модель
        model_path = f'models/best_{model_name}_model.joblib'
        joblib.dump(model, model_path)
        
        # Сохраняем порог
        threshold_path = f'models/best_{model_name}_threshold.joblib'
        joblib.dump(threshold, threshold_path)
        
        print(f"\nЛучшая модель ({model_name}) сохранена:")
        print(f"- Модель: {model_path}")
        print(f"- Порог: {threshold_path}")
        
        # Сохраняем информацию о признаках
        features_path = f'models/best_{model_name}_features.joblib'
        joblib.dump(self.selected_features, features_path)
        print(f"- Признаки: {features_path}")

    def run_pipeline(self, df):
        """
        Запуск полного пайплайна
        
        Parameters:
        -----------
        df : pd.DataFrame
            Исходный датасет
            
        Returns:
        --------
        results : dict
            Словарь с результатами:
            - selected_features: list, отобранные признаки
            - best_models: dict, лучшие модели
            - optimal_thresholds: dict, оптимальные пороги
            - best_model_name: str, название лучшей модели
            - best_score: float, лучший ROC-AUC score
            - report_path: str, путь к сохраненному отчету
        """
        # Предобработка
        df = self.preprocess_data(df)
        
        # Отбор признаков
        self.select_best_features(df)
        
        # Оптимизация моделей
        self.optimize_models(df)
        
        # Выбор лучшей модели на основе ROC-AUC на тестовой выборке
        test_data = df[df['group'] == 'test']
        X_test = test_data[self.selected_features]
        y_test = test_data[self.config['target_column']]
        
        best_model_name = None
        best_score = -1
        
        for model_name, model in self.best_models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
            print(f"ROC-AUC для {model_name}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        print(f"\nВыбрана лучшая модель: {best_model_name} (ROC-AUC: {best_score:.4f})")
        
        # Оставляем только лучшую модель
        best_model = self.best_models[best_model_name]
        self.best_models = {best_model_name: best_model}
        
        # Оптимизация порога только для лучшей модели
        print(f"\nОптимизация порога для модели ({best_model_name})...")
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Находим оптимальный порог
        threshold, metrics = find_optimal_threshold(
            y_test,
            y_pred_proba,
            method='hybrid',
            target_recall=self.config['threshold_config'].get('target_recall')
        )
        
        self.optimal_thresholds = {best_model_name: threshold}
        
        # Визуализация
        plot_threshold_analysis(y_test, y_pred_proba, threshold)
        
        # Анализ влияния порога
        results_df = pd.DataFrame({
            'target': y_test,
            'score': y_pred_proba
        })
        analyze_threshold_impact(results_df, threshold)
        
        # Сохранение лучшей модели
        self.save_best_model(
            best_model_name,
            best_model,
            threshold
        )
        
        # Анализ лучшей модели и создание отчета
        report_path = self.analyze_model(df, best_model_name)
        
        print("\n=== Итоги работы пайплайна ===")
        print(f"Лучшая модель: {best_model_name}")
        print(f"ROC-AUC: {best_score:.4f}")
        print(f"Оптимальный порог: {threshold:.3f}")
        print(f"Отчет сохранен: {report_path}")
        
        return {
            'selected_features': self.selected_features,
            'best_models': self.best_models,
            'optimal_thresholds': self.optimal_thresholds,
            'best_model_name': best_model_name,
            'best_score': best_score,
            'report_path': report_path
        }

def load_data(file_path, target_column, features=None, exclude_columns=None):
    """
    Загрузка и валидация данных
    
    Parameters:
    -----------
    file_path : str
        Путь к файлу с данными (CSV)
    target_column : str
        Название целевой переменной
    features : list, optional
        Список названий признаков для использования. Если указан, `exclude_columns` игнорируется.
    exclude_columns : list, optional
        Список колонок, которые нужно исключить из анализа (используется, только если `features` не задан).
        
    Returns:
    --------
    df : pd.DataFrame
        Загруженный датафрейм
    final_features : list
        Итоговый список признаков для использования в пайплайне
    """
    try:
        df = pd.read_csv(file_path)
        all_columns = df.columns.tolist()

        if target_column not in all_columns:
             raise ValueError(f"Целевая переменная '{target_column}' не найдена в файле {file_path}")
        
        if features is not None:
            # Сценарий 2: Используем предоставленный список признаков
            print("Используются явно указанные признаки.")
            # Проверка предоставленных признаков
            missing_provided = [col for col in features if col not in all_columns]
            if missing_provided:
                 raise ValueError(f"Следующие указанные признаки отсутствуют в данных: {missing_provided}")
            final_features = features[:] # Создаем копию
            # Убедимся, что целевая переменная не в списке признаков
            if target_column in final_features:
                print(f"Предупреждение: Целевая переменная '{target_column}' была удалена из списка признаков.")
                final_features.remove(target_column)
        else:
            # Сценарий 1 или 3: Определяем признаки автоматически
            potential_features = all_columns[:]
            potential_features.remove(target_column)
            
            if exclude_columns is not None:
                # Сценарий 3: Используем все, кроме целевой и исключенных колонок
                print("Используются все признаки, кроме целевой переменной и исключенных столбцов.")
                final_features = [col for col in potential_features if col not in exclude_columns]
                excluded_but_not_found = [col for col in exclude_columns if col not in all_columns]
                if excluded_but_not_found:
                     print(f"Предупреждение: Следующие столбцы для исключения не найдены в данных: {excluded_but_not_found}")
            else:
                # Сценарий 1: Используем все колонки, кроме целевой
                print("Используются все признаки, кроме целевой переменной.")
                final_features = potential_features

        # Финальные проверки
        if not final_features:
             raise ValueError("Список признаков для анализа пуст.")
             
        required_columns = final_features + [target_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            # Эта проверка может быть избыточной из-за предыдущих, но полезна для надежности
            raise ValueError(f"В данных отсутствуют необходимые колонки: {missing_columns}")

        if df[target_column].isnull().any():
            raise ValueError("В целевой переменной есть пропущенные значения")
            
        unique_values = df[target_column].unique()
        if len(unique_values) != 2:
             raise ValueError(f"Целевая переменная должна быть бинарной. Найдены значения: {unique_values}")

        print(f"\nДанные успешно загружены и проверены.")
        
        # Возвращаем только необходимые колонки для возможной экономии памяти
        return df[required_columns], final_features
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Файл {file_path} пуст")
    except Exception as e:
        raise Exception(f"Ошибка при загрузке или проверке данных: {str(e)}")

def prepare_pipeline(df, config):
    """
    Подготовка пайплайна с использованием предоставленной конфигурации.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм с данными (уже должен содержать только нужные колонки)
    config : dict
        Конфигурация для настройки пайплайна. Должна содержать ключ 'features'
        с актуальным списком признаков.
        
    Returns:
    --------
    pipeline : CreditScoringPipeline
        Настроенный пайплайн для обучения
    """
    # Проверка наличия целевой переменной (уже сделана в load_data, но для надежности)
    if config['target_column'] not in df.columns:
        raise ValueError(f"Целевая переменная '{config['target_column']}' не найдена в данных")
    
    # Проверка наличия списка признаков в конфигурации
    if 'features' not in config or not config['features']:
         raise ValueError("Конфигурация должна содержать непустой список признаков ('features')")

    # Проверка, что все признаки из конфига есть в датафрейме
    missing_features = [f for f in config['features'] if f not in df.columns]
    if missing_features:
        raise ValueError(f"Следующие признаки из конфигурации отсутствуют в переданном DataFrame: {missing_features}")

    # Выводим информацию о данных, с которыми будет работать пайплайн
    print("\nПодготовка пайплайна:")
    print(f"- Количество строк: {len(df)}")
    print(f"- Количество используемых признаков: {len(config['features'])}")
    
    # Визуализация распределения целевой переменной
    plt.figure(figsize=(8, 6))
    target_dist = df[config['target_column']].value_counts(normalize=True)
    target_dist.plot(kind='bar')
    plt.title('Распределение целевой переменной')
    plt.xlabel('Класс')
    plt.ylabel('Доля наблюдений')
    plt.xticks([0, 1], ['Хороший заемщик', 'Плохой заемщик'], rotation=0)
    
    # Добавляем значения над столбцами
    for i, v in enumerate(target_dist):
        plt.text(i, v, f'{v:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return CreditScoringPipeline(config)

def main():
    """
    Основная функция для запуска пайплайна.
    """
    file_path = 'data_pd.csv'  # Укажите ваш путь к файлу
    target_column = 'default_12m'  # Укажите вашу целевую переменную

    # --- Варианты определения признаков ---
    # Вариант 1: Автоматически (все столбцы кроме target_column)
    explicit_features = None
    exclude_cols = None 
    
    # Вариант 2: Явно указать признаки
    # explicit_features = ['feature_1', 'feature_4', 'feature_10', 'feature_15'] 
    # exclude_cols = None # Будет проигнорирован, если explicit_features задан

    # Вариант 3: Указать столбцы для исключения
    # explicit_features = None
    # exclude_cols = ['id_column', 'another_useless_column'] 

    # -----------------------------------------

    try:
        # Шаг 1: Загрузка и валидация данных, получение актуального списка признаков
        df, actual_features = load_data(
            file_path, 
            target_column=target_column,
            features=explicit_features,
            exclude_columns=exclude_cols
        )

        # Шаг 2: Подготовка конфигурации пайплайна с актуальными признаками
        pipeline_config = {
            'random_state': PREPROCESSING_CONFIG['random_state'],
            'features': actual_features,  # Используем актуальный список признаков
            'target_column': target_column,
            'feature_selection': FEATURE_SELECTION_CONFIG,
            'model_config': MODEL_CONFIG,
            'threshold_config': THRESHOLD_CONFIG,
            'explainability_config': EXPLAINABILITY_CONFIG
            # Добавляем сюда исходные списки категориальных/числовых признаков из config.py,
            # если они нужны для prepare_data. prepare_data должен будет сам отфильтровать
            # те из них, что попали в actual_features.
            # 'categorical_features_config': PREPROCESSING_CONFIG.get('categorical_features', []),
            # 'numerical_features_config': PREPROCESSING_CONFIG.get('numerical_features', [])
        }
        
        # Передаем в prepare_data только те категориальные/числовые признаки из конфига, 
        # которые реально присутствуют в actual_features
        pipeline_config['categorical_features'] = [f for f in PREPROCESSING_CONFIG.get('categorical_features', []) if f in actual_features]
        pipeline_config['numerical_features'] = [f for f in PREPROCESSING_CONFIG.get('numerical_features', []) if f in actual_features]


        pipeline = prepare_pipeline(df, pipeline_config)

        # Шаг 3: Запуск пайплайна
        results = pipeline.run_pipeline(df) # df уже содержит только нужные колонки

        # Вывод результатов
        print("\n--- Результаты работы пайплайна ---")
        print(f"Отобранные признаки ({len(results['selected_features'])}): {results['selected_features']}")
        print(f"Лучшая модель: {results['best_model_name']} (ROC-AUC: {results['best_score']:.4f})")
        print("Оптимальные пороги:", results['optimal_thresholds'])

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"\nОшибка выполнения пайплайна: {e}")

if __name__ == "__main__":
    main() 