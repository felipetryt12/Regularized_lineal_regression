
# your code here
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm

# Configurar pandas para mostrar todas las columnas
pd.set_option('display.max_columns', None)

# Función para cargar datos
def load_data(url):
    try:
        data = pd.read_csv(url)
        print("Datos cargados exitosamente.")
        return data
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

# Función para guardar datos
def save_data(data, file_path):
    try:
        data.to_csv(file_path, index=False)
        print(f'Archivo guardado en: {file_path}')
    except Exception as e:
        print(f"Error al guardar el archivo: {e}")

# Función para mostrar información básica del dataset
def explore_data(data):
    print(data.shape)
    print(data.head(3))
    print(data.info())

# Función para imprimir nombres de columnas
def print_column_names(data):
    for column in data.columns:
        print(column)

# Función para mostrar las primeras filas de columnas categóricas
def show_categorical_head(data):
    print(data.select_dtypes("object").head(3))

# Función para mostrar la distribución de una variable numérica
def plot_numeric_distributions(data, numeric_vars):
    numero_de_columnas = 3
    numero_de_filas = math.ceil(len(numeric_vars)/numero_de_columnas)
    
    fig, axes = plt.subplots(nrows=numero_de_filas, ncols=numero_de_columnas, figsize=(15, 5*numero_de_filas))
    axes = axes.flatten()
    
    for ax, col in zip(axes, numeric_vars):
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        
    for i in range(len(numeric_vars), len(axes)):
        fig.delaxes(axes[i])
        
    plt.tight_layout()
    plt.show()

# Función para calcular y mostrar un heatmap de correlaciones
def plot_correlation_heatmap(data, features, title):
    corr = data[features].corr()
    plt.figure(figsize=(15,10))
    sns.heatmap(corr, annot=True, cmap="rocket_r")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.title(title)
    plt.show()

# Función para realizar regresión lineal simple y mostrar resultados
def simple_linear_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Error cuadrático medio (MSE): {mse}')
    print(f'R²: {r2}')
    return model

# Función para encontrar el mejor umbral de correlación
def find_optimal_correlation_threshold(data, target, thresholds, model):
    results = []

    if target not in data.columns:
        raise ValueError(f"La columna objetivo '{target}' no se encuentra en el DataFrame")

    X = data.drop(columns=[target])
    y = data[target]
    X_numeric = X.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    correlation_matrix = pd.DataFrame(X_scaled, columns=X_numeric.columns).corr().abs()

    for threshold in thresholds:
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        X_reduced = X_numeric.drop(columns=to_drop)
        scores = cross_val_score(model, X_reduced, y, cv=5, scoring='neg_mean_squared_error')
        results.append((threshold, -scores.mean(), len(X_reduced.columns)))

    return results

# Función para ajustar y evaluar un modelo con ElasticNet
def elastic_net_regression(X_train, y_train, X_test, y_test):
    elastic_net = ElasticNet()
    grid_param = {
        'alpha': [.1, 1, 10],
        'l1_ratio': [0, .2, .4, .6, .8, 1.0]
    }
    grid_search = GridSearchCV(estimator=elastic_net, param_grid=grid_param, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_param = grid_search.best_params_

    print(f"Mejores parámetros: {best_param}")
    good_en = ElasticNet(alpha=best_param['alpha'], l1_ratio=best_param['l1_ratio'])
    good_en.fit(X_train, y_train)

    y_pred = good_en.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Elastic Net MSE: {mse}')
    print(f'Elastic Net R²: {r2}')
    return good_en

# Cargar datos
url = "https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv"
data_raw = load_data(url)

# Guardar datos en local
file_path = rf'C:\Users\wipip\OneDrive\Documentos\GitHub\Regularized_lineal_regression-main\data\raw\datos_cruditos.csv'
save_data(data_raw, file_path)

# Exploración inicial de datos
explore_data(data_raw)
print_column_names(data_raw)
show_categorical_head(data_raw)

# Variables numéricas
numeric_vars = data_raw.select_dtypes(include=['int64', 'float64']).columns
plot_numeric_distributions(data_raw, numeric_vars)

# Analizar correlaciones
recursos_feat = [ 'Active Physicians per 100000 Population 2018 (AAMC)', 'Total Active Patient Care Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active Patient Care Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active General Surgeons per 100000 Population 2018 (AAMC)', 'Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)', 'Total nurse practitioners (2019)', 'Total physician assistants (2019)', 'Total Hospitals (2019)', 'Internal Medicine Primary Care (2019)', 'Family Medicine/General Practice Primary Care (2019)', 'Total Specialist Physicians (2019)', 'ICU Beds_x']
plot_correlation_heatmap(data_raw, recursos_feat, 'Correlación entre recursos de salud')

# Regresión lineal simple
pobreza_feat = 'PCTPOVALL_2018'
condiciones_feat = ['Obesity_prevalence', 'Heart disease_prevalence', 'Active Physicians per 100000 Population 2018 (AAMC)']
X_poverty = data_raw[[pobreza_feat]]
y_health = data_raw['Active Physicians per 100000 Population 2018 (AAMC)']
simple_linear_regression(X_poverty, y_health)

# Encontrar el mejor umbral de correlación
thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
results = find_optimal_correlation_threshold(data_raw, 'Active Physicians per 100000 Population 2018 (AAMC)', thresholds, Ridge())
for threshold, score, num_features in results:
    print(f'Umbral: {threshold}, Error cuadrático medio: {score}, Número de características: {num_features}')

# Graficar resultados de la búsqueda del umbral óptimo
thresholds, scores, num_features = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, scores, marker='o', label='MSE')
plt.xlabel('Umbral de Correlación')
plt.ylabel('Error Cuadrático Medio')
plt.title('Rendimiento del Modelo vs. Umbral de Correlación')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(thresholds, num_features, marker='o', label='Número de Características', color='orange')
plt.xlabel('Umbral de Correlación')
plt.ylabel('Número de Características')
plt.title('Número de Características vs. Umbral de Correlación')
plt.legend()
plt.grid(True)
plt.show()

# Preparar datos para el modelo
data_p_df = data_raw[[f'fips', 'TOT_POP', '0-9 y/o % of total pop', '10-19 y/o % of total pop', '20-29 y/o % of total pop', '30-39 y/o % of total pop', '40-49 y/o % of total pop', '50-59 y/o % of total pop', '60-69 y/o % of total pop', '70-79 y/o % of total pop', '80+ y/o % of total pop', '% White-alone', 'Black-alone pop', '% Black-alone', 'Native American/American Indian-alone pop', '% NA/AI-alone', 'Asian-alone pop', '% Asian-alone', 'Hawaiian/Pacific Islander-alone pop', '% Hawaiian/PI-alone', 'Two or more races pop', '% Two or more races', 'N_POP_CHG_2018', 'GQ_ESTIMATES_2018', 'R_birth_2018', 'R_death_2018', 'R_NATURAL_INC_2018', 'R_INTERNATIONAL_MIG_2018', 'R_DOMESTIC_MIG_2018', 'Percent of adults with less than a high school diploma 2014-18', 'Percent of adults with a high school diploma only 2014-18', "Percent of adults completing some college or associate's degree 2014-18", "Percent of adults with a bachelor's degree or higher 2014-18", 'PCTPOVALL_2018', 'PCTPOV017_2018', 'MEDHHINC_2018', 'Unemployment_rate_2018', 'Med_HH_Income_Percent_of_State_Total_2018', 'Active Physicians per 100000 Population 2018 (AAMC)', 'Active Primary Care Physicians per 100000 Population 2018 (AAMC)', 'Active General Surgeons per 100000 Population 2018 (AAMC)', 'Active Patient Care General Surgeons per 100000 Population 2018 (AAMC)', 'Total Hospitals (2019)', 'ICU Beds_x', 'Percent of Population Aged 60+', 'STATE_FIPS', 'CNTY_FIPS', 'anycondition_prevalence', 'Obesity_prevalence', 'Heart disease_prevalence', 'COPD_prevalence', 'diabetes_prevalence', 'CKD_prevalence', 'Urban_rural_code']].copy()

# Guardar datos procesados
file_path_processed = rf'C:\Users\wipip\OneDrive\Documentos\GitHub\Regularized_lineal_regression-main\data\processed\datos_procesados.csv'
save_data(data_p_df, file_path_processed)

# Modelo de regresión lineal y evaluación
X = data_p_df.drop('Active Physicians per 100000 Population 2018 (AAMC)', axis=1)
y = data_p_df['Active Physicians per 100000 Population 2018 (AAMC)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelo de ElasticNet
elastic_net_model = elastic_net_regression(X_train_scaled, y_train, X_test_scaled, y_test)

# Comparación con otros modelos
def compare_models(X_train, y_train, X_test, y_test):
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'{name} MSE: {mse}')
        print(f'{name} R²: {r2}')
        
compare_models(X_train_scaled, y_train, X_test_scaled, y_test)
