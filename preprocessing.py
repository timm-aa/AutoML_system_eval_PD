"""
Модуль для предобработки данных
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict


def is_numeric_string(text):
    """Проверка, является ли строка числовой"""
    if not isinstance(text, str): return False
    try:
        float(text)
        return True
    except ValueError:
        return False

def detect_feature_type(series, nunique_threshold=20, text_len_threshold=5):
    """Определение типа признака"""
    # Проверка на полностью пропущенные значения
    if series.isnull().all(): return 'categorical' # Или 'drop'/'unknown'
    # Проверка на числовой тип (int, float, bool)
    if pd.api.types.is_numeric_dtype(series.dropna()): return 'numerical'
    if pd.api.types.is_bool_dtype(series.dropna()): return 'numerical'
    # Работаем с нечисловыми типами
    series_str = series.dropna().astype(str)
    if series_str.empty: return 'categorical'
    # # Проверка на ОКВЭД
    # try:
    #     if series_str.apply(is_okved).all(): return 'okved'
    # except Exception: pass
    # Проверка на числовые строки
    try:
        if series_str.apply(is_numeric_string).all(): return 'numerical'
    except Exception: pass
    # Проверка на текст
    try:
        mean_len = series_str.str.len().mean()
        nunique = series_str.nunique()
        if mean_len > text_len_threshold and nunique > nunique_threshold: return 'text'
    except Exception: pass
    # По умолчанию - категориальный
    return 'categorical'

# --- SmartFeatureTransformer Class (Moved to top level) ---
class SmartFeatureTransformer:
    def __init__(self, text_max_features=100, nunique_threshold=20, text_len_threshold=5, missing_threshold=0.9):
        self.feature_types = {}
        self.scalers = {}
        self.encoders = {} # Для OHE
        self.vectorizers = {} # Для TF-IDF
        self._output_feature_names = None
        self.features_to_keep = [] # Список признаков, прошедших порог пропусков
        self.text_max_features = text_max_features
        self.nunique_threshold = nunique_threshold
        self.text_len_threshold = text_len_threshold
        self.missing_threshold = missing_threshold # Порог доли пропусков для удаления

    def fit(self, X, y=None):
        """
        Подгонка трансформера: удаление признаков с пропусками, определение типов,
        обучение кодировщиков/масштабировщиков.
        """
        print("Подгонка Трансформера...")

        # 1. Удаление признаков с долей пропусков > missing_threshold
        missing_ratios = X.isnull().mean()
        self.features_to_keep = missing_ratios[missing_ratios <= self.missing_threshold].index.tolist()
        features_to_drop = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()

        if features_to_drop:
            print(f"Удалены признаки с долей пропусков > {self.missing_threshold:.0%}: {features_to_drop}")
        if not self.features_to_keep:
                raise ValueError("Не осталось признаков после удаления по порогу пропусков.")

        X_kept = X[self.features_to_keep]
        self.feature_types = {col: detect_feature_type(X_kept[col], self.nunique_threshold, self.text_len_threshold) for col in X_kept.columns}
        print("Определены типы признаков.")

        # 3. Обучение
        for col, f_type in self.feature_types.items():
            if f_type == 'numerical':
                scaler = MinMaxScaler()
                scaler.fit(X_kept[[col]]) # fit на данных с возможными NaN

                # Проверка scale_ на ноль
                if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    # Проверяем, есть ли нули в scale_ (для DataFrame это массив)
                    zero_scale_mask = (scaler.scale_ == 0)
                    if np.any(zero_scale_mask):
                        print(f"Предупреждение: Признак '{col}' имеет нулевую дисперсию. Заменяем масштаб на 1.")
                        # Создаем копию scale_ чтобы изменить его
                        new_scale = scaler.scale_.copy()
                        new_scale[zero_scale_mask] = 1.0 # Заменяем нули на 1
                        scaler.scale_ = new_scale # Присваиваем измененный массив

                self.scalers[col] = scaler

            elif f_type == 'categorical':
                feature_data_str = X_kept[[col]].astype(str)
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                self.encoders[col] = encoder.fit(feature_data_str)
            elif f_type == 'text':
                feature_data_str = X_kept[col].astype(str)
                vectorizer = TfidfVectorizer(max_features=self.text_max_features)
                self.vectorizers[col] = vectorizer.fit(feature_data_str)

        self._generate_output_feature_names(X_kept.columns)
        print(f"Ожидается {len(self._output_feature_names)} выходных признаков.")
        return self

    def transform(self, X):
        """Применение обученных трансформаций."""
        if not self.features_to_keep:
            raise RuntimeError("Трансформер не был обучен или не осталось признаков. Вызовите fit() перед transform().")

        X_kept = X[self.features_to_keep]
        transformed_cols = []

        for col in X_kept.columns:
            f_type = self.feature_types.get(col)
            if f_type is None: continue
            feature_data = X_kept[[col]]

            if f_type == 'numerical':
                scaler = self.scalers.get(col)
                if scaler:
                    scaled_data = scaler.transform(feature_data)
                    transformed_cols.append(scaled_data)
                else:
                    transformed_cols.append(feature_data.values)
            elif f_type == 'categorical':
                encoder = self.encoders.get(col)
                if encoder:
                    onehot_data = encoder.transform(feature_data.astype(str))
                    transformed_cols.append(onehot_data)
            elif f_type == 'text':
                vectorizer = self.vectorizers.get(col)
                if vectorizer:
                    tfidf_data = vectorizer.transform(feature_data[col].astype(str)).toarray()
                    transformed_cols.append(tfidf_data)

        if not transformed_cols:
            print("Предупреждение: Ни один признак не был трансформирован.")
            return np.empty((X.shape[0], 0))

        transformed_array = np.hstack(transformed_cols)

        # Проверка и коррекция размерности
        expected_cols = len(self._output_feature_names)
        actual_cols = transformed_array.shape[1]
        
        if expected_cols != actual_cols:
            print(f"Коррекция размерности признаков: ожидалось {expected_cols}, получено {actual_cols}.")
            if actual_cols < expected_cols:
                # Добавляем столбцы с нулями для недостающих признаков
                padding = np.zeros((transformed_array.shape[0], expected_cols - actual_cols))
                transformed_array = np.hstack([transformed_array, padding])
                print(f"Добавлено {expected_cols - actual_cols} нулевых столбцов.")
            elif actual_cols > expected_cols:
                # Обрезаем лишние столбцы (редкий случай)
                print(f"Обрезано {actual_cols - expected_cols} лишних столбцов.")
                transformed_array = transformed_array[:, :expected_cols]

        # Дополнительная проверка для уверенности
        if transformed_array.shape[1] != expected_cols:
            print(f"ОШИБКА: После коррекции размерность всё равно не соответствует! Получено {transformed_array.shape[1]}, ожидалось {expected_cols}")
        
        return transformed_array

    def fit_transform(self, X, y=None):
        """Подгонка и трансформация за один шаг."""
        self.fit(X, y)
        return self.transform(X)

    def _generate_output_feature_names(self, input_features_kept):
        """Генерация имен выходных признаков после обучения."""
        output_names = []
        for col in input_features_kept:
            f_type = self.feature_types.get(col)
            if f_type is None: continue

            if f_type == 'numerical':
                output_names.append(f"{col}_num_scaled")
            elif f_type == 'categorical':
                encoder = self.encoders.get(col)
                if encoder:
                    # Получаем имена из OneHotEncoder
                    ohe_names = encoder.get_feature_names_out([col])
                    
                    # Очищаем имена от спецсимволов (для моделей)
                    cleaned_names = []
                    for name in ohe_names:
                        # Заменяем '[', ']' и другие спецсимволы на '_'
                        cleaned_name = name.replace('[', '_').replace(']', '_')
                        # Добавим еще замены для других спецсимволов JSON
                        cleaned_name = cleaned_name.replace('"', '_').replace("'", '_')
                        cleaned_name = cleaned_name.replace(':', '_').replace(',', '_')
                        cleaned_name = cleaned_name.replace('{', '_').replace('}', '_')
                        cleaned_names.append(cleaned_name)
                    
                    output_names.extend(cleaned_names)
                else: 
                    output_names.append(f"{col}_cat_failed")
            elif f_type == 'text':
                vectorizer = self.vectorizers.get(col)
                if vectorizer:
                    # Получаем имена из TfidfVectorizer
                    tfidf_names = vectorizer.get_feature_names_out([col])
                    
                    # НОВОЕ: Очищаем имена от спецсимволов
                    cleaned_names = []
                    for name in tfidf_names:
                        # Такая же очистка, как и для категориальных
                        cleaned_name = name.replace('[', '_').replace(']', '_')
                        cleaned_name = cleaned_name.replace('"', '_').replace("'", '_')
                        cleaned_name = cleaned_name.replace(':', '_').replace(',', '_')
                        cleaned_name = cleaned_name.replace('{', '_').replace('}', '_')
                        cleaned_names.append(cleaned_name)
                    
                    output_names.extend(cleaned_names)
                else: 
                    output_names.append(f"{col}_text_failed")
        
        self._output_feature_names = output_names

    def get_feature_names_out(self, input_features=None):
        """Возвращает имена признаков после трансформации."""
        if self._output_feature_names is None:
                raise RuntimeError("Имена выходных признаков недоступны. Вызовите fit() сначала.")
        return self._output_feature_names

# --- create_preprocessing_pipeline Function (Now just uses the class) ---
def create_preprocessing_pipeline(features):
    """
    Создание экземпляра SmartFeatureTransformer (обертка для совместимости, если нужна).
    """
    # Эта функция больше не нужна, так как prepare_data использует класс напрямую.
    # Оставляем ее пока для обратной совместимости, если она где-то вызывается.
    print("Предупреждение: create_preprocessing_pipeline вызывается, но prepare_data использует SmartFeatureTransformer напрямую.")
    # Возвращаем экземпляр класса, но без подгонки
    return SmartFeatureTransformer()


# --- prepare_data Function (Uses the top-level class) ---
def prepare_data(df, target_column, config):
    """
    Подготовка данных: разделение, подгонка и применение препроцессора (без импутации).
    """
    print("Подготовка данных...")
    df_copy = df.copy()
    initial_features = config['features']

    # Проверка и замена inf
    numeric_cols = df_copy.select_dtypes(include=np.number).columns
    # Проверяем только в исходных признаках, чтобы не трогать таргет случайно
    cols_to_check = [col for col in initial_features if col in numeric_cols]
    if cols_to_check: # Если есть числовые признаки для проверки
        inf_mask = df_copy[cols_to_check].isin([np.inf, -np.inf])
        if inf_mask.any().any(): # Проверяем, есть ли хоть одно True во всей маске
            cols_with_inf = inf_mask.any()[inf_mask.any()].index.tolist()
            print(f"Предупреждение: Обнаружены бесконечные значения в колонках: {cols_with_inf}. Заменяем на NaN.")
            df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 1. Разделение на train/test/duplicates
    feature_cols_for_duplicates = initial_features
    duplicates = df_copy.duplicated(subset=feature_cols_for_duplicates, keep='first')
    df_copy['group'] = 'train'
    df_copy.loc[duplicates, 'group'] = 'duplicate'
    non_duplicate_idx = df_copy[~duplicates].index
    train_idx, test_idx = [], []
    if len(non_duplicate_idx) >= 2:
        try:
            train_idx, test_idx = train_test_split(
                non_duplicate_idx, test_size=0.2, random_state=config['random_state'],
                stratify=df_copy.loc[non_duplicate_idx, target_column]
            )
        except ValueError as e:
            print(f"Предупреждение: Стратифицированное разделение не удалось ({e}). Используется обычное.")
            if len(non_duplicate_idx) >= 2:
                 train_idx, test_idx = train_test_split(
                     non_duplicate_idx, test_size=0.2, random_state=config['random_state']
                 )
    df_copy.loc[train_idx, 'group'] = 'train'
    df_copy.loc[test_idx, 'group'] = 'test'
    if len(non_duplicate_idx) < 2: df_copy.loc[non_duplicate_idx, 'group'] = 'train'
    print(f"Распределение по группам:\n{df_copy['group'].value_counts()}")

    # 2. Создание и подгонка препроцессора (SmartFeatureTransformer)
    X = df_copy[initial_features]
    X_train_for_missing_check = df_copy.loc[train_idx, initial_features]

    if X_train_for_missing_check.empty:
         raise ValueError("Нет данных в обучающей выборке для проверки пропусков.")

    preprocessing_params = config.get('preprocessing_params', {})
    # Используем класс напрямую
    preprocessor = SmartFeatureTransformer(**preprocessing_params)

    # Подгоняем на X_train
    preprocessor.fit(X_train_for_missing_check)

    # 3. Применение препроцессора ко всем данным (X)
    print("Применение препроцессора ко всем данным...")
    processed_array = preprocessor.transform(X)

    # 4. Получение имен обработанных признаков
    processed_feature_names = preprocessor.get_feature_names_out()

    # 5. Создание DataFrame с обработанными признаками
    processed_features_df = pd.DataFrame(
        processed_array,
        columns=processed_feature_names,
        index=X.index
    )

    # ---> НАЧАЛО ИЗМЕНЕНИЯ: Проверка inf/large values ПОСЛЕ трансформации <---
    # Иногда проблемы могут возникнуть и после трансформации
    numeric_processed_cols = processed_features_df.select_dtypes(include=np.number).columns
    if not numeric_processed_cols.empty:
        inf_mask_processed = processed_features_df[numeric_processed_cols].isin([np.inf, -np.inf])
        if inf_mask_processed.any().any():
            cols_with_inf_processed = inf_mask_processed.any()[inf_mask_processed.any()].index.tolist()
            print(f"Предупреждение: Обнаружены бесконечные значения ПОСЛЕ трансформации в колонках: {cols_with_inf_processed}. Заменяем на NaN.")
            # Заменяем прямо в processed_features_df
            processed_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Дополнительно проверим на очень большие значения (опционально)
        # large_value_threshold = np.finfo(np.float64).max / 10  # Пример порога
        # large_mask = np.abs(processed_features_df[numeric_processed_cols]) > large_value_threshold
        # if large_mask.any().any():
        #     cols_with_large = large_mask.any()[large_mask.any()].index.tolist()
        #     print(f"Предупреждение: Обнаружены очень большие значения ПОСЛЕ трансформации в колонках: {cols_with_large}. Проверьте данные/трансформации.")
            # Замена на NaN или другое значение, если необходимо
            # processed_features_df[large_mask] = np.nan
    # ---> КОНЕЦ ИЗМЕНЕНИЯ <---

    # 6. Объединение с 'group' и 'target_column'
    final_df = pd.concat([
        processed_features_df,
        df_copy[['group', target_column]]
        ], axis=1)

    print(f"Подготовка данных завершена. Размер обработанных данных: {final_df.shape}")
    return final_df, list(processed_features_df.columns)