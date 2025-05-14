import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.feature_selection import f_classif, SelectKBest
import json
import os

# Definición de variables globales
target_column = 'ciudad'
inferencia = []
columns_to_drop = ["evento_desc","evento_categoria","reddit_temp","reddit_hum","reddit_desc","airport_code","origin_iata","estimated_price_usd"]
categorical_to_numerical = [{'categorical_col': 'desc_actual'}, {'categorical_col': 'temporada'}, 
                            {'categorical_col': 'perfil_viajero'}, {'categorical_col': 'entornos'}, 
                            {'categorical_col': 'clasificacion_destino'}, {'categorical_col': 'airline'}
                            , {'categorical_col': 'stops'}, {'categorical_col': 'class'},] 

def explore_data(df):
    """1. Exploración de Datos."""
    print("Información general del dataframe:")
    df.info()
    print("\nEstadísticas descriptivas:")
    print(df.shape)
    return df

def clean_duplicates(df):
    """1.1 Quitar Duplicados."""
    df.drop_duplicates(inplace=True)
    print(f"Registros duplicados eliminados: {len(df) - len(df.drop_duplicates())}")
    return df

def clean_irrelevant_data(df):
    """1.2 Eliminar información irrelevante."""
    df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')
    print(f"Columnas irrelevantes eliminadas: {columns_to_drop}")
    return df


def univariate_categorical_analysis(df):
    """Análisis univariante de variables categóricas (top 20)."""
    filtered_df = df.copy()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    num_cols = len(categorical_cols)
    cols_per_row = 3
    num_rows = (num_cols + cols_per_row - 1) // cols_per_row

    fig, axs = plt.subplots(num_rows, cols_per_row, figsize=(15, num_rows * 4))
    axs = axs.flatten()

    for i, col in enumerate(categorical_cols):
        top_values = df[col].value_counts().nlargest(20).index
        filtered_df = df[df[col].isin(top_values)].copy()

        filtered_df[col] = filtered_df[col].apply(lambda x: x[:25] + '...' if isinstance(x, str) and len(x) > 25 else x)
        
        sns.countplot(
            data=filtered_df,
            x=col,
            order=filtered_df[col].value_counts().index,
            ax=axs[i],
            hue=col,  # Se usa hue para aplicar la paleta
            palette='viridis',
            legend=False,  # Evita mostrar leyenda redundante
            dodge=False
        )
        axs[i].set_title(f'Distribución de {col}', fontsize=10)
        axs[i].tick_params(axis='x', labelrotation=45, labelsize=8)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')
        axs[i].margins(x=0.01)  # Reduce el espacio entre barras
    for label in axs[i].get_xticklabels():
        label.set_ha('right')

    for j in range(i + 1, len(axs) if i + 1 < len(axs) else i + 1):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()



def univariate_numerical_analysis(df):
    """2.2 Análisis de variables numéricas."""
    # Condición añadida: Factorizar target_column si es categórico y actualizar target_column
    global target_column
    if df[target_column].dtype == 'object':
        df[target_column + '_n'] = pd.factorize(df[target_column])[0]
        transformation_rules = {row[target_column]: row[target_column + '_n'] for _, row in df[[target_column, target_column + '_n']].drop_duplicates().iterrows()}
        ruta_json = os.path.join("../data/processed/Json", f"{target_column}_transformation_rules.json")
        os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
        with open(ruta_json, "w") as f:
            json.dump(transformation_rules, f)
        target_column = target_column + '_n'  # Actualizar target_column
    
    numerical_cols = df.select_dtypes(include=['number']).columns.difference([target_column])
    num_numerical = len(numerical_cols)
    num_rows = (num_numerical + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.histplot(x=col, data=df, kde=True, ax=axes[i])
        axes[i].set_title(f'Distribución de {col}')
        if i + num_numerical < len(axes):
            sns.boxplot(y=col, data=df, ax=axes[i + num_numerical])
            axes[i + num_numerical].set_title(f'Boxplot de {col}')
    for i in range(num_numerical * 2, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout()
    plt.show()

def bivariate_numerical_analysis(df):
    """3.1 Análisis numérico-numérico."""
    numerical_cols = df.select_dtypes(include=['number']).columns.difference([target_column])
    if len(numerical_cols) > 1:
        num_plots = len(numerical_cols) * (len(numerical_cols) - 1) // 2
        cols = 3
        rows = (num_plots // cols) + (1 if num_plots % cols != 0 else 0)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        plot_index = 0
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                ax = axes[plot_index]
                sns.scatterplot(x=numerical_cols[i], y=numerical_cols[j], data=df, ax=ax)
                ax.set_title(f'{numerical_cols[i]} vs {numerical_cols[j]}')
                plot_index += 1
        for k in range(plot_index, len(axes)):
            fig.delaxes(axes[k])
        plt.tight_layout()
        plt.show()

def bivariate_categorical_analysis(df):
    """Análisis bivariante de variables categóricas (top 5 por variable), con visualización en filas de 3 gráficos."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    from itertools import combinations
    import warnings

    sns.set(style="whitegrid")
    warnings.filterwarnings("ignore", category=UserWarning)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_pairs = list(combinations(categorical_cols, 2))

    valid_plots = []

    for col1, col2 in categorical_pairs:
        top1 = df[col1].value_counts().nlargest(5).index
        top2 = df[col2].value_counts().nlargest(5).index
        filtered_df = df[df[col1].isin(top1) & df[col2].isin(top2)]

        if not filtered_df.empty:
            valid_plots.append((col1, col2, filtered_df, top1, top2))

    # Mostrar los gráficos en filas de 3
    for i in range(0, len(valid_plots), 3):
        subset = valid_plots[i:i+3]
        fig, axes = plt.subplots(1, len(subset), figsize=(6 * len(subset), 5))
        if len(subset) == 1:
            axes = [axes]
        for ax, (col1, col2, data, top1, top2) in zip(axes, subset):
            sns.countplot(data=data, x=col1, hue=col2, order=top1, hue_order=top2, ax=ax, palette="Set2")
            ax.set_title(f'{col1} por {col2}')
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.show()
        plt.close(fig)

def class_predictor_analysis(df):
    """Gráficos tipo boxplot entre variables numéricas y la primera categórica (top 5 categorías)."""
    numerical_cols = df.select_dtypes(include='number').columns.difference([target_column])
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if not categorical_cols.empty:
        cat_col = categorical_cols[0]
        top_cat = df[cat_col].value_counts().nlargest(5).index
        filtered_df = df[df[cat_col].isin(top_cat)]

        num_cols = list(numerical_cols)
        n = len(num_cols)
        cols_per_row = 4
        rows = (n + cols_per_row - 1) // cols_per_row

        fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 5 * rows))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            sns.boxplot(data=filtered_df, x=cat_col, y=col, order=top_cat, color='mediumseagreen', ax=axes[i])
            axes[i].set_title(f'{col} por {cat_col}')
            axes[i].tick_params(axis='x', rotation=45)
        
        # Oculta ejes vacíos si hay menos de 4*n gráficos
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()

    else:
        pass  # No imprimir nada si no hay columnas categóricas

def correlation_analysis(df):
    """3.4 Análisis de correlaciones."""
    if categorical_to_numerical:
        for conversion in categorical_to_numerical:
            categorical_col = conversion['categorical_col']
            numerical_col = conversion.get('numerical_col', f"{categorical_col}_n")
            df[numerical_col] = pd.factorize(df[categorical_col])[0]
            transformation_rules = {
                row[categorical_col]: row[numerical_col]
                for _, row in df[[categorical_col, numerical_col]].drop_duplicates().iterrows()
            }
            ruta_json = os.path.join("../data/processed/Json", f"{numerical_col}_transformation_rules.json")
            os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
            with open(ruta_json, "w") as f:
                json.dump(transformation_rules, f)
    numerical_df = df.select_dtypes(include='number')
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        numerical_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="Set2",  # Podés cambiar por "vlag", "coolwarm", etc.
        annot_kws={"size": 8},
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.title('Matriz de correlación', fontsize=14)
    plt.tight_layout()
    plt.show()
    

def categorical_numerical_correlation(df):
    """Boxplots entre categóricas (top 5 valores) y variables numéricas."""
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = df.select_dtypes(include='number').columns
    if len(categorical_cols) == 0 or len(numerical_cols) == 0:
        print("No hay suficientes columnas numéricas y/o categóricas para generar los gráficos de correlación.")
        return
    for cat_col in categorical_cols:
        top_vals = df[cat_col].value_counts().nlargest(5).index
        filtered_df = df[df[cat_col].isin(top_vals)]
        n = len(numerical_cols)
        cols = 2
        rows = (n + 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
        axes = axes.flatten()
        for i, num_col in enumerate(numerical_cols):
            sns.boxplot(
                data=filtered_df,
                x=cat_col,
                y=num_col,
                order=top_vals,
                palette="Set2",
                ax=axes[i]
            )
            axes[i].set_title(f'{num_col} por {cat_col}')
            axes[i].tick_params(axis='x', rotation=45)
        # Eliminar subgráficos vacíos si sobran
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout()
        plt.suptitle(f'Boxplots de variables numéricas por {cat_col}', fontsize=16, y=1.02)
        plt.show()

def pairplot_analysis(df):
    """4. Análisis de toda la data en una."""
    sns.pairplot(df)
    plt.show()

def analyze_outliers(df):
    """5.1 Análisis Outliers."""
    df_con_outliers = df.copy()
    df_sin_outliers = df.copy()
    numerical_cols = df.select_dtypes(include=['number']).columns.difference([target_column])
    # Paleta de colores (puede ampliarse o repetirse si hay más columnas)
    colores = sns.color_palette("Set2", n_colors=len(numerical_cols))
    num_cols = len(numerical_cols)
    rows = (num_cols + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, 5 * rows))
    axes = axes.flatten()
    for i, col in enumerate(numerical_cols):
        sns.boxplot(ax=axes[i], data=df, y=col, color=colores[i % len(colores)])
        axes[i].set_title(col)
    for j in range(num_cols, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
    return df_sin_outliers, numerical_cols

def replace_outliers(df_sin_outliers, numerical_cols):
    """Reemplazar outliers."""
    def replace_outliers_column(column, df):
        colum_stats = df[column].describe()
        IQR = colum_stats["75%"] - colum_stats["25%"]
        lower_bound = colum_stats["25%"] - 1.5 * IQR
        upper_bound = colum_stats["75%"] + 1.5 * IQR
        if lower_bound < 0: lower_bound = min(df[column])
        df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else x)
        df[column] = df[column].apply(lambda x: lower_bound if x < lower_bound else x)
        return df.copy(), [lower_bound, upper_bound]

    outliers_dict = {}
    for column in numerical_cols:
        df_sin_outliers, limit_list = replace_outliers_column(column, df_sin_outliers)
        outliers_dict[column] = [float(limit) for limit in limit_list]
    ruta_json = os.path.join("../data/processed/Json", "outliers_dict.json")
    os.makedirs(os.path.dirname(ruta_json), exist_ok=True) # Crea el directorio si no existe
    with open(ruta_json, "w") as f:
        json.dump(outliers_dict, f)
    print(outliers_dict)
    return df_sin_outliers

def handle_missing_values(df_sin_outliers):
    """5.2 Análisis de valores faltantes."""
    print("Valores faltantes por columna:")
    print(df_sin_outliers.isnull().sum())
    numerical_cols = df_sin_outliers.select_dtypes(include=['number']).columns.difference([target_column])
    categorical_cols = df_sin_outliers.select_dtypes(include=['object', 'category']).columns
    for col in numerical_cols:
        df_sin_outliers[col] = df_sin_outliers[col].fillna(df_sin_outliers[col].median())
    for col in categorical_cols:
        df_sin_outliers[col] = df_sin_outliers[col].fillna(df_sin_outliers[col].mode()[0])
    print("\nValores faltantes después de la imputación:")
    print(df_sin_outliers.isnull().sum())
    return df_sin_outliers

def infer_new_features(df_sin_outliers):
    """5.3 Inferencia de nuevas características."""
    numerical_cols = df_sin_outliers.select_dtypes(include=['number']).columns.difference([target_column])
    if len(numerical_cols) >= 2:
        for feature in inferencia:
            try:
                df_sin_outliers[feature['new_col_name']] = df_sin_outliers[feature['col1']] * df_sin_outliers[feature['col2']]
                print(f"Nueva característica '{feature['new_col_name']}' creada a partir de '{feature['col1']}' y '{feature['col2']}'.")
            except Exception as e:
                print(f"Error al crear la nueva característica '{feature['new_col_name']}': {e}")
    else:
        print("No hay columnas que apliquen para la inferencia.")
    return df_sin_outliers

def feature_scaling(df, df_sin_outliers, ruta_guardado="../data/processed/X&Ys"):
    """6. Feature Scalling."""
    numerical_cols = df.select_dtypes(include=['number']).columns.difference([target_column])
    X_con_outliers = df.drop(target_column, axis=1)[numerical_cols]
    X_sin_outliers = df_sin_outliers.drop(target_column, axis=1)[numerical_cols]
    y = df[target_column]
    X_train_con_outliers, X_test_con_outliers, y_train, y_test = train_test_split(X_con_outliers, y, test_size=0.2, random_state=42)
    X_train_sin_outliers, X_test_sin_outliers = train_test_split(X_sin_outliers, test_size=0.2, random_state=42)
    X_train_con_outliers.to_excel(os.path.join(ruta_guardado, "X_train_con_outliers.xlsx"), index=False)
    X_train_sin_outliers.to_excel(os.path.join(ruta_guardado, "X_train_sin_outliers.xlsx"), index=False)
    X_test_con_outliers.to_excel(os.path.join(ruta_guardado, "X_test_con_outliers.xlsx"), index=False)
    X_test_sin_outliers.to_excel(os.path.join(ruta_guardado, "X_test_sin_outliers.xlsx"), index=False)
    y_train.to_excel(os.path.join(ruta_guardado, "y_train.xlsx"), index=False)
    y_test.to_excel(os.path.join(ruta_guardado, "y_test.xlsx"), index=False)
    print("Archivos creados: X_train_con_outliers.xlsx, X_train_sin_outliers.xlsx, X_test_con_outliers.xlsx, X_test_sin_outliers.xlsx, y_train.xlsx, y_test.xlsx")
    return X_train_con_outliers, X_test_con_outliers, X_train_sin_outliers, X_test_sin_outliers, y_train, y_test, numerical_cols

def normalize_data(X_train_con_outliers, X_test_con_outliers, X_train_sin_outliers, X_test_sin_outliers, numerical_cols, ruta_guardado="../data/processed/X&Ys", ruta_modelo="../models/"):
    """6.1 Normalización."""
    normalizador_con_outliers = StandardScaler()
    normalizador_con_outliers.fit(X_train_con_outliers)
    with open(os.path.join(ruta_modelo, "normalizador_con_outliers.pkl"), "wb") as file:
        pickle.dump(normalizador_con_outliers, file)
    X_train_con_outliers_norm = normalizador_con_outliers.transform(X_train_con_outliers)
    X_train_con_outliers_norm = pd.DataFrame(X_train_con_outliers_norm, index=X_train_con_outliers.index, columns=numerical_cols)
    X_test_con_outliers_norm = normalizador_con_outliers.transform(X_test_con_outliers)
    X_test_con_outliers_norm = pd.DataFrame(X_test_con_outliers_norm, index=X_test_con_outliers.index, columns=numerical_cols)
    X_train_con_outliers_norm.to_excel(os.path.join(ruta_guardado, "X_train_con_outliers_norm.xlsx"), index=False)
    X_test_con_outliers_norm.to_excel(os.path.join(ruta_guardado, "X_test_con_outliers_norm.xlsx"), index=False)
    normalizador_sin_outliers = StandardScaler()
    normalizador_sin_outliers.fit(X_train_sin_outliers)
    with open(os.path.join(ruta_modelo, "normalizador_sin_outliers.pkl"), "wb") as file:
        pickle.dump(normalizador_sin_outliers, file)
    X_train_sin_outliers_norm = normalizador_sin_outliers.transform(X_train_sin_outliers)
    X_train_sin_outliers_norm = pd.DataFrame(X_train_sin_outliers_norm, index=X_train_sin_outliers.index, columns=numerical_cols)
    X_test_sin_outliers_norm = normalizador_sin_outliers.transform(X_test_sin_outliers)
    X_test_sin_outliers_norm = pd.DataFrame(X_test_sin_outliers_norm, index=X_test_sin_outliers.index, columns=numerical_cols)
    X_train_sin_outliers_norm.to_excel(os.path.join(ruta_guardado, "X_train_sin_outliers_norm.xlsx"), index=False)
    X_test_sin_outliers_norm.to_excel(os.path.join(ruta_guardado, "X_test_sin_outliers_norm.xlsx"), index=False)
    print("Archivos creados: X_train_con_outliers_norm.xlsx, X_test_con_outliers_norm.xlsx, X_train_sin_outliers_norm.xlsx, X_test_sin_outliers_norm.xlsx")
    return X_train_con_outliers_norm, X_test_con_outliers_norm, X_train_sin_outliers_norm, X_test_sin_outliers_norm

def scale_min_max_data_1(X_train_con_outliers, X_test_con_outliers, X_train_sin_outliers, X_test_sin_outliers, numerical_cols, ruta_guardado="../data/processed/X&Ys", ruta_modelo="../models/"):
    """
    Escala los DataFrames, guarda los scalers entrenados y los resultados en archivos XLSX.

    Args:
        X_train_con_outliers (pd.DataFrame): DataFrame de entrenamiento con outliers.
        X_test_con_outliers (pd.DataFrame): DataFrame de prueba con outliers.
        X_train_sin_outliers (pd.DataFrame): DataFrame de entrenamiento sin outliers.
        X_test_sin_outliers (pd.DataFrame): DataFrame de prueba sin outliers.
        numerical_cols (list): Lista de columnas numéricas a escalar.
        ruta_guardado (str): Ruta donde guardar los archivos XLSX.
        ruta_modelo (str): Ruta donde guardar los modelos scaler.

    Returns:
        tuple: Tupla con los cuatro DataFrames escalados.
    """
    try:
        # Asegurar que la carpeta del modelo exista
        os.makedirs(ruta_modelo, exist_ok=True)
        os.makedirs(ruta_guardado, exist_ok=True)

        # Escalar con MinMaxScaler (con outliers)
        scaler_con_outliers = MinMaxScaler()
        scaler_con_outliers.fit(X_train_con_outliers[numerical_cols])

        with open(os.path.join(ruta_modelo, "scaler_con_outliers.pkl"), "wb") as file:
            pickle.dump(scaler_con_outliers, file)

        X_train_con_outliers_scaled = X_train_con_outliers.copy()
        X_test_con_outliers_scaled = X_test_con_outliers.copy()

        X_train_con_outliers_scaled[numerical_cols] = scaler_con_outliers.transform(X_train_con_outliers[numerical_cols])
        X_test_con_outliers_scaled[numerical_cols] = scaler_con_outliers.transform(X_test_con_outliers[numerical_cols])

        # Escalar con StandardScaler (sin outliers)
        scaler_sin_outliers = StandardScaler()
        scaler_sin_outliers.fit(X_train_sin_outliers[numerical_cols])

        with open(os.path.join(ruta_modelo, "scaler_sin_outliers.pkl"), "wb") as file:
            pickle.dump(scaler_sin_outliers, file)

        X_train_sin_outliers_scaled = X_train_sin_outliers.copy()
        X_test_sin_outliers_scaled = X_test_sin_outliers.copy()

        X_train_sin_outliers_scaled[numerical_cols] = scaler_sin_outliers.transform(X_train_sin_outliers[numerical_cols])
        X_test_sin_outliers_scaled[numerical_cols] = scaler_sin_outliers.transform(X_test_sin_outliers[numerical_cols])

        # Guardar los DataFrames escalados en archivos XLSX
        X_train_con_outliers_scaled.to_excel(os.path.join(ruta_guardado, "X_train_con_outliers_scal.xlsx"), index=False)
        X_test_con_outliers_scaled.to_excel(os.path.join(ruta_guardado, "X_test_con_outliers_scal.xlsx"), index=False)
        X_train_sin_outliers_scaled.to_excel(os.path.join(ruta_guardado, "X_train_sin_outliers_scal.xlsx"), index=False)
        X_test_sin_outliers_scaled.to_excel(os.path.join(ruta_guardado, "X_test_sin_outliers_scal.xlsx"), index=False)

        print("DataFrames escalados, modelos guardados y archivos XLSX creados.")
        return X_train_con_outliers_scaled, X_test_con_outliers_scaled, X_train_sin_outliers_scaled, X_test_sin_outliers_scaled

    except Exception as e:
        print(f"Error en scale_min_max_data: {e}")
        return None, None, None, None

def feature_selection(X_train_con_outliers, X_test_con_outliers, X_train_sin_outliers, X_test_sin_outliers, y_train, y_test, target_column, ruta_modelo="../models/"):
    """7. Feature Selection."""
    try:
        feature_selection_k = int(input("Ingrese el valor de k para la selección de características: "))
        dataset_name = input("Ingrese el nombre del dataset para entrenar el modelo (X_train_con_outliers, X_train_sin_outliers, X_test_con_outliers, X_test_sin_outliers, X_train_con_outliers_norm, X_train_sin_outliers_norm, X_test_con_outliers_norm, X_test_sin_outliers_norm, X_train_con_outliers_scal, X_train_sin_outliers_scal, X_test_con_outliers_scal, X_test_sin_outliers_scal): ")
        
        if dataset_name == "X_train_con_outliers":
            feature_selection_dataset = X_train_con_outliers
        elif dataset_name == "X_train_sin_outliers":
            feature_selection_dataset = X_train_sin_outliers
        elif dataset_name == "X_test_con_outliers":
            feature_selection_dataset = X_test_con_outliers
        elif dataset_name == "X_test_sin_outliers":
            feature_selection_dataset = X_test_sin_outliers
        else:
            raise ValueError("Nombre de dataset no válido.")
    except ValueError as e:
        print(f"Error: {e}")
        return None, None

    modelo_seleccion = SelectKBest(f_classif, k=feature_selection_k)
    modelo_seleccion.fit(feature_selection_dataset, y_train)
    ix = modelo_seleccion.get_support()
    
    x_train_sel = pd.DataFrame(modelo_seleccion.transform(feature_selection_dataset), columns=feature_selection_dataset.columns.values[ix])
    
    # Seleccionamos el test dataset a usar dependiendo del dataset de entrenamiento.
    if dataset_name == "X_train_con_outliers":
        x_test_sel = pd.DataFrame(modelo_seleccion.transform(X_test_con_outliers), columns=X_test_con_outliers.columns.values[ix])
    elif dataset_name == "X_train_sin_outliers":
        x_test_sel = pd.DataFrame(modelo_seleccion.transform(X_test_sin_outliers), columns=X_test_sin_outliers.columns.values[ix])
    
    
    x_train_sel[target_column] = list(y_train)
    x_test_sel[target_column] = list(y_test)
    ruta_json = os.path.join("../data/processed/Json", f"featureselection_k_{feature_selection_k}.json")
    os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
    with open(ruta_json, "w") as f:
        json.dump(list(x_train_sel.columns), f)
    x_train_sel.to_csv(os.path.join(ruta_modelo, "x_train_sel.csv"), index=False)
    x_test_sel.to_csv(os.path.join(ruta_modelo, "x_test_sel.csv"), index=False)
    x_train_sel.to_excel(os.path.join(ruta_modelo, "x_train_sel.xlsx"), index=False) # Guarda x_train_sel en XLSX
    x_test_sel.to_excel(os.path.join(ruta_modelo, "x_test_sel.xlsx"), index=False) # Guarda x_test_sel en XLSX
    print(f"Características seleccionadas: {list(x_train_sel.columns)}")
    return x_train_sel, x_test_sel