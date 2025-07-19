import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List, Dict, Union

def identify_features(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    """
    Ідентифікує числові та категоріальні колонки у DataFrame.

    Видаляє 'CustomerId' та 'Surname', оскільки вони не є інформативними ознаками,
    а також 'id', якщо він є, і цільову колонку.

    Args:
        df (pd.DataFrame): Вхідний DataFrame.
        target_column (str): Назва цільової колонки.

    Returns:
        Tuple[List[str], List[str]]: Кортеж зі списками числових та категоріальних колонок.
    """
    # Виключаємо колонки, які не використовуються як ознаки
    exclude_cols = ['id', 'CustomerId', 'Surname', target_column]
    features_df = df.drop(columns=[col for col in exclude_cols if col in df.columns], errors='ignore')

    numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = features_df.select_dtypes(include='object').columns.tolist()
    
    return numeric_cols, categorical_cols

def scale_numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    scaler: Union[StandardScaler, None] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Масштабує числові ознаки за допомогою StandardScaler.

    Args:
        df (pd.DataFrame): Вхідний DataFrame.
        numeric_cols (List[str]): Список назв числових колонок для масштабування.
        scaler (Union[StandardScaler, None]): Навчений StandardScaler або None.
                                               Якщо None, буде створено новий скейлер.
        fit (bool): Якщо True, скейлер буде навчено на `df`. Якщо False, буде використано
                    існуючий скейлер для трансформації.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Кортеж з DataFrame з масштабованими ознаками
                                          та навченим або використаним StandardScaler.
    """
    if not numeric_cols:
        return df, scaler # Якщо немає числових колонок, повертаємо оригінальний DF та None/існуючий скейлер

    df_copy = df.copy()
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    else:
        df_copy[numeric_cols] = scaler.transform(df_copy[numeric_cols])

    return df_copy, scaler

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoder: Union[OneHotEncoder, None] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, OneHotEncoder, List[str]]:
    """
    Кодує категоріальні ознаки за допомогою OneHotEncoder.

    Args:
        df (pd.DataFrame): Вхідний DataFrame.
        categorical_cols (List[str]): Список назв категоріальних колонок для кодування.
        encoder (Union[OneHotEncoder, None]): Навчений OneHotEncoder або None.
                                               Якщо None, буде створено новий енкодер.
        fit (bool): Якщо True, енкодер буде навчено на `df`. Якщо False, буде використано
                    існуючий енкодер для трансформації.

    Returns:
        Tuple[pd.DataFrame, OneHotEncoder, List[str]]: Кортеж з DataFrame з закодованими ознаками,
                                                 навченим або використаним OneHotEncoder,
                                                 та списком назв нових закодованих колонок.
    """
    if not categorical_cols:
        return df, encoder, [] # Якщо немає категоріальних колонок

    df_copy = df.copy()
    if encoder is None:
        # handle_unknown='ignore' дозволяє обробляти категорії, яких немає у тренувальному наборі
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse_output=False для повернення щільного масиву

    if fit:
        encoded_data = encoder.fit_transform(df_copy[categorical_cols])
    else:
        encoded_data = encoder.transform(df_copy[categorical_cols])

    # Створюємо DataFrame з новими закодованими колонками
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df_copy.index)

    # Об'єднуємо закодовані ознаки з рештою DataFrame (числовими та іншими)
    df_processed = pd.concat([df_copy.drop(columns=categorical_cols), encoded_df], axis=1)

    return df_processed, encoder, list(encoder.get_feature_names_out(categorical_cols))

def preprocess_data(
    raw_df: pd.DataFrame,
    target_column: str = 'Exited',
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_numeric: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Union[StandardScaler, None], OneHotEncoder]:
    """
    Виконує повну попередню обробку даних для навчання моделі.

    Включає розбиття на тренувальний та валідаційний набори,
    обробку категоріальних даних (One-Hot Encoding) та опціональне масштабування числових даних.

    Args:
        raw_df (pd.DataFrame): Сирий вхідний DataFrame.
        target_column (str): Назва цільової колонки.
        test_size (float): Частка даних для валідаційного набору.
        random_state (int): Зерно для відтворюваності розбиття даних.
        scaler_numeric (bool): Якщо True, числові ознаки будуть масштабовані.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Union[StandardScaler, None], OneHotEncoder]:
            - X_train: Тренувальний набір ознак.
            - train_targets: Цільова змінна для тренувального набору.
            - X_val: Валідаційний набір ознак.
            - val_targets: Цільова змінна для валідаційного набору.
            - input_cols: Перелік назв всіх колонок-ознак, які використовуються в X_train/X_val.
            - scaler: Навчений StandardScaler (або None, якщо scaler_numeric=False).
            - encoder: Навчений OneHotEncoder.
    """
    # Визначаємо вхідні ознаки та цільову змінну
    X = raw_df.drop(columns=[col for col in ['id', 'CustomerId', 'Surname', target_column] if col in raw_df.columns], errors='ignore')
    y = raw_df[target_column]

    # Розбиття на тренувальний та валідаційний набори
    X_train_raw, X_val_raw, train_targets, val_targets = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Ідентифікація числових та категоріальних колонок (на основі тренувальних даних)
    numeric_cols, categorical_cols = identify_features(X_train_raw, target_column)

    # Обробка категоріальних даних (навчаємо енкодер на тренувальних даних)
    X_train_encoded, encoder, new_cat_cols = encode_categorical_features(X_train_raw, categorical_cols, fit=True)
    X_val_encoded, _, _ = encode_categorical_features(X_val_raw, categorical_cols, encoder=encoder, fit=False) # Використовуємо навчений енкодер

    # Масштабування числових даних (опціонально)
    scaler = None
    if scaler_numeric:
        X_train_scaled, scaler = scale_numeric_features(X_train_encoded, numeric_cols, fit=True)
        X_val_scaled, _ = scale_numeric_features(X_val_encoded, numeric_cols, scaler=scaler, fit=False)
    else:
        X_train_scaled = X_train_encoded
        X_val_scaled = X_val_encoded
    
    # Визначаємо фінальний список колонок (важливо для X_train, X_val, X_new)
    # Порядок колонок може змінитися після OneHotEncoding, тому оновлюємо список
    final_numeric_cols = [col for col in numeric_cols if col in X_train_scaled.columns] # Ті, що залишились після можливого виключення
    input_cols = final_numeric_cols + new_cat_cols # Об'єднуємо числові та нові закодовані категоріальні

    # Переконаємося, що X_train та X_val містять лише `input_cols` і в правильному порядку
    X_train_processed = X_train_scaled[input_cols]
    X_val_processed = X_val_scaled[input_cols]

    return X_train_processed, train_targets, X_val_processed, val_targets, input_cols, scaler, encoder


def preprocess_new_data(
    new_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Union[StandardScaler, None],
    encoder: OneHotEncoder,
    input_cols: List[str], # Фінальний список колонок, які очікує модель
    scaler_numeric: bool = True
) -> pd.DataFrame:
    """
    Обробляє нові (тестові/небачені) дані, використовуючи вже навчені трансформери.

    Args:
        new_df (pd.DataFrame): Новий вхідний DataFrame (наприклад, test.csv).
        numeric_cols (List[str]): Список назв числових колонок, визначених під час навчання.
        categorical_cols (List[str]): Список назв категоріальних колонок, визначених під час навчання.
        scaler (Union[StandardScaler, None]): Навчений StandardScaler (або None, якщо не використовувався).
        encoder (OneHotEncoder): Навчений OneHotEncoder.
        input_cols (List[str]): Список назв всіх колонок-ознак, які очікує модель.
                                 Використовується для вирівнювання колонок.
        scaler_numeric (bool): Чи використовувалося масштабування числових ознак під час навчання.

    Returns:
        pd.DataFrame: Оброблені дані, готові для передбачення моделлю.
    """
    # Видаляємо колонки, які не є ознаками, якщо вони присутні в new_df
    X_new_raw = new_df.drop(columns=[col for col in ['id', 'CustomerId', 'Surname'] if col in new_df.columns], errors='ignore')

    # Обробка категоріальних даних
    X_new_encoded, _, _ = encode_categorical_features(X_new_raw, categorical_cols, encoder=encoder, fit=False)

    # Масштабування числових даних (якщо використовувалося під час навчання)
    if scaler_numeric and scaler is not None:
        X_new_scaled, _ = scale_numeric_features(X_new_encoded, numeric_cols, scaler=scaler, fit=False)
    else:
        X_new_scaled = X_new_encoded
    
    # Вирівнювання колонок: переконуємось, що X_new_processed має ті ж колонки і в тому ж порядку
    #, що й X_train/X_val
    X_new_processed = X_new_scaled.reindex(columns=input_cols, fill_value=0)

    return X_new_processed

# Додамо імпорт numpy для identify_features, якщо його ще немає
import numpy as np