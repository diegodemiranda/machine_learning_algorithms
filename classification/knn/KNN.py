import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# Lê um arquivo CSV especificado por file_path usando a biblioteca pandas.
# O dataset está balanceado e possui 1996 resultados 'bad' e 2004 'good'.
def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found.")
    except Exception as e:
        print("Error: ", e)
        return None


def treat_dataset(df):
    # Realiza o pré-processamento dos dados.
    # Mapeamento dos valores 'good' e 'bad' para 1 e 0, respectivamente.
    df['Quality'] = df['Quality'].replace({'good': 1, 'bad': 0})
    
    # Remoção de linhas com valores ausentes
    df.dropna(inplace=True)
    
    # Conversão dos tipos de dados das colunas 'Acidity' e 'Quality'.
    df['Acidity'] = df['Acidity'].astype(float)
    df['Quality'] = df['Quality'].astype(int)
    
    df.info()
    
    return df


# Recebe um DataFrame df contendo os dados e um parâmetro k para o algoritmo k-NN.
def knn_tests(df, k):
    # Divide os dados em atributos (variáveis independentes) x e rótulos (variável dependente) y.
    x = df.drop(columns=['A_id', 'Quality'])
    y = df['Quality']
    
    # Normaliza os dados no intervalo [0, 1] com a normalização Min-Max.
    x = (x - x.min()) / (x.max() - x.min())
    normal = x.join(y)
    print("normalized matrix:")
    print(normal)
    
    # Calcula o tamanho do conjunto de dados e o divide em dados de treinamento e teste.
    # O resultado é convertido para um número inteiro usando a função int().
    
    train_size = int((80 / 100) * len(df))
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Implementa o algoritmo k-NN para fazer previsões no conjunto de teste. Essas previsões são baseadas nos
    # rótulos dos k vizinhos mais próximos para cada amostra no conjunto de teste, utilizando a maioria de
    # votos para determinar a classe prevista.
    predictions = []
    # Inicia um loop que percorre cada amostra no conjunto de teste.
    for i in range(len(x_test)):
        # Calcula as distâncias entre a amostra atual do conjunto de teste e todas as amostras no conjunto de
        # treinamento. Isso é feito usando a distância euclidiana, que é a raiz quadrada da soma dos quadrados das
        # diferenças entre os atributos de cada amostra.
        distances = np.sqrt(np.sum((x_train - x_test.iloc[i]) ** 2, axis=1))
        # Identifica os índices das k amostras mais próximas no conjunto de treinamento, com base nas menores
        # distâncias calculadas. Isso é feito ordenando as distâncias e selecionando os índices dos k primeiros
        # elementos do array ordenado.
        nearest_neighbors = distances.argsort()[:k]
        # Obtém os rótulos correspondentes às amostras mais próximas encontradas no conjunto de treinamento.
        nearest_labels = y_train.iloc[nearest_neighbors]
        # Determina o rótulo mais comum entre os vizinhos mais próximos. Isso é feito calculando a moda dos rótulos.
        most_common_label = nearest_labels.mode()[0]
        # Adiciona a previsão (rótulo mais comum) para a amostra atual à lista de previsões.
        predictions.append(most_common_label)
    
    # Calcula e escreve a matriz de confusão e os resultados obtidos para cada métrica.
    print("Confusion matrix:")
    conf_matrix = confusion_matrix(y_test, predictions)
    # Calcula a precisão do modelo, que é a proporção de todas as previsões corretas
    # (verdadeiros positivos e verdadeiros negativos) em relação ao número total de amostras.
    accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / (
            conf_matrix[0][0] + conf_matrix[1][1] + conf_matrix[1][0] + conf_matrix[0][1])
    
    # Calcula a precisão, que é a proporção de verdadeiros positivos em relação ao total de previsões positivas
    # (verdadeiros positivos mais falsos positivos). Isso mede a capacidade do modelo de classificar
    # corretamente as amostras como positivas.
    precision = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
    
    # Calcula a sensibilidade, também conhecida como recall, que é a proporção de verdadeiros positivos em relação ao
    # total de amostras verdadeiramente positivas (verdadeiros positivos mais falsos negativos).
    # Isso mede a capacidade do modelo de identificar corretamente todas as amostras positivas.
    recall = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    
    # Calcula a especificidade, que é a proporção de verdadeiros negativos em relação ao total de amostras
    # verdadeiramente negativas (verdadeiros negativos mais falsos positivos). Isso mede a capacidade do modelo
    # de identificar corretamente todas as amostras negativas.
    specificity = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
    
    # recall == valor preditivo positivo         sensibilidade == precision
    measuref = (2 * (precision * recall)) / (precision + recall)
    
    print(conf_matrix)
    print("Accuracy:", ((predictions == y_test).mean() * 100), "%")
    print("Precision: ", precision * 100, " %")
    print("Recall: ", recall * 100, " %")
    print("Specificity: ", specificity * 100, " %")
    print("Measure-F: ", measuref * 100, " %")
    
    # Retorna as previsões feitas pelo modelo.
    return predictions, conf_matrix, accuracy, precision, recall, specificity, measuref


def call_knn_tests_with_column_exclusion(df, k):
    # Obtém todas as colunas, removendo as colunas 'A_id' e 'Quality' da lista.
    columns_to_exclude = df.columns.difference(['A_id', 'Quality'])
    
    results = []
    # Inicia um loop que itera sobre cada coluna a ser excluída do DataFrame.
    for column in columns_to_exclude:
        print("\nExcluindo a coluna:", column)
        # Chama a função knn_tests que executa os testes sem a coluna atualmente excluída.
        predictions, conf_matrix, accuracy, precision, recall, specificity, measuref = (
            knn_tests(df.drop(columns=[column]), k))
        # Adiciona os resultados dos testes à lista results, incluindo o nome da coluna excluída,
        # a matriz de confusão e descompacta a tupla retornada pela função em várias variáveis individuais.
        results.append((column, conf_matrix, accuracy, precision, recall, specificity, measuref))
    # Abre um arquivo de texto chamado 'knn_results.txt' em modo de escrita e escreve as informações a seguir para
    # cada iteração.
    with open('knn_results.txt', 'w') as file:
        for column, conf_matrix, accuracy, precision, recall, specificity, measuref in results:
            file.write(f"Column removed: ({column}) \n")
            file.write("Confusion matrix:\n")
            for row in conf_matrix:
                file.write(f"{row}\n")
            file.write(f"Accuracy: {accuracy * 100} %\n")
            file.write(f"Precision: {precision * 100} %\n")
            file.write(f"Recall: {recall * 100} %\n")
            file.write(f"Specificity: {specificity * 100} %\n")
            file.write(f"Measure-F: {measuref * 100} %\n")
            file.write("\n")
    
    # Abre o arquivo 'knn_results.txt' usando o aplicativo padrão do sistema operacional.
    os.system('open knn_results.txt')


def plot_conf_matrix(conf_matrix):
    # Plota a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Samples")
    plt.show()


# Define o caminho do arquivo CSV contendo os dados das maçãs. Chama a função read_csv_file() para ler os dados.
def main():
    file_path = 'apple_quality.csv'
    apples = read_csv_file(file_path)
    k = 11
    # Se os dados forem lidos com sucesso, chama a função treat_dataset() para pré-processá-los e, em
    # seguida, chama a função knn_tests() para realizar o teste com o algoritmo KNN.
    if apples is not None:
        apples = treat_dataset(apples)
        predictions, conf_matrix, accuracy, precision, recall, specificity, measuref = knn_tests(apples, k)
        call_knn_tests_with_column_exclusion(apples, k)
        plot_conf_matrix(conf_matrix)


# Garante que o código dentro deste bloco só será executado se o script for executado diretamente e não importado
# como um módulo em outro script. Isso é uma boa prática em Python para modularizar o código e evitar que
# blocos de código sejam executados acidentalmente ao importar um módulo.
if __name__ == "__main__":
    main()
