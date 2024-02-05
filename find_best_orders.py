
from plot import plot
from pmdarima import arima
import pandas as pd
import numpy as np
import os
import re

# def divide_xy(series, lag=4):
#   x = []
#   y = []
#   for i in series[:lag]:
#     x.append(series[i:i+lag])
#     y.append(series[i+4])
#   return x, y

# Funções de cálculo de erro
def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

def mse(y, y_hat):
    return np.mean(np.square(y - y_hat))

def rmse(y, y_hat):
    return np.sqrt(np.mean(np.square(y - y_hat)))

# def mape(y, y_hat):
#     return np.mean(np.abs((y - y_hat) / y) * 100)

# Caminhos pros dados
# Mudar esse . de acordo com o diretório da pasta dos dados
data_folder = '.'
sub_folders = ['less_malwares', 'more_malwares']
data_lengths = ['data_one_year', 'data_six_months', 'data_two_years']

dataframe = {
  'nome': [],
  'modelo': [],
  'mae_treino': [],
  'mae_teste': [],
  'mse_treino': [],
  'mse_teste': [],
  'rmse_treino': [],
  'rmse_teste': [],
}
# Iterando por todos os arquivos de dados
for sub_folder in sub_folders:
  for provider in os.listdir(f'{data_folder}/{sub_folder}'):
    for data_length in data_lengths:
      # Pegando os caminhos e nome dos arquivos de treino e teste
      files = os.listdir(f'{data_folder}/{sub_folder}/{provider}/{data_length}')
      files = list(filter(lambda file: file[-4:] == '.csv', files))
      training_file = [file for file in files if re.search('training', file)][0]
      training_file = f'{data_folder}/{sub_folder}/{provider}/{data_length}/{training_file}'
      test_file = [file for file in files if re.search('test', file)][0]
      test_file = f'{data_folder}/{sub_folder}/{provider}/{data_length}/{test_file}'

      # Abrindo o arquivo de treino, renomeando coluna e passando pra array
      series = pd.read_csv(training_file)
      series.columns = [series.columns[0], 'semana', *series.columns[2:]]
      prediction_series = np.array(series['qtde'])
      
      # Procurando a ordem do arima ideal pra série, retorna todos os fits válidos
      models = arima.auto_arima(prediction_series, return_valid_fits=True)

      # Abrindo séries de teste
      test_series = pd.read_csv(test_file)
      test_series.columns = [test_series.columns[0], 'semana', *test_series.columns[2:]]
      test_series = np.array(test_series['qtde'])

      # Itera pelos três melhores fits, faz as predições e calcula os erros de treino e teste
      for model in models[:3]:
        predictions = model.predict_in_sample()
        mae_treino = mae(prediction_series, predictions)
        mse_treino = mse(prediction_series, predictions)
        rmse_treino = rmse(prediction_series, predictions)
        test_predictions = []
        for leitura in test_series:
          test_predictions.append(model.predict(1)[0])
          model.update([leitura])
        mae_teste = mae(test_series, test_predictions)
        mse_teste = mse(test_series, test_predictions)
        rmse_teste = rmse(test_series, test_predictions)
        plot(np.concatenate((prediction_series, test_series)), divisions=[len(prediction_series)], sec_plots=[np.concatenate((predictions, test_predictions))], save=True, show=False, img_name=f'./imagens/{sub_folder}_{provider}_{data_length}_{str(model)}.jpeg', )
        # Adiciona os valores relevantes num dataframe
        dataframe['nome'].append(f'{sub_folder}_{provider}_{data_length}')
        dataframe['modelo'].append(str(model))
        dataframe['mae_treino'].append(f'{mae_treino:.3f}')
        dataframe['mae_teste'].append(f'{mae_teste:.3f}')
        dataframe['mse_treino'].append(f'{mse_treino:.3f}')
        dataframe['mse_teste'].append(f'{mse_teste:.3f}')
        dataframe['rmse_treino'].append(f'{rmse_treino:.3f}')
        dataframe['rmse_teste'].append(f'{rmse_teste:.3f}')


# Salva o DataFrame
pd.DataFrame(dataframe).to_csv('./resultado.csv', index=False)
