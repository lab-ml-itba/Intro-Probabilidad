# Implementar función que grafique y devuelva la posición del robot luego de N iteraciones

import numpy as np
from matplotlib import pyplot as plt

def plot_bar_chart(P, indexes=None, index_str='$P(S=%s)$' ,title='Función de masa de probabilidad de la posición del robot: $P(S=k)$'):
    N = len(P)
    if indexes is None:
        indexes = np.linspace(1, N, N, dtype = int) # Desde, Hasta, Cantidad, Tipo
    plt.figure(figsize=(20, 5)) # Tamaño del gráfico
    plt.bar(indexes, P, width=0.75, color='b') # Grafico
    plt.title(title)

    # Definición de indices
    string_indexes = [index_str%i for i in indexes]
    plt.xlim([0,N+1])
    plt.xticks(indexes, string_indexes) 
    plt.xticks(rotation=60)
    plt.show()

def update_hist(likelihood, prior):
    posterior_un = likelihood*prior
    posterior = posterior_un/posterior_un.sum()
    return posterior

def take_step(P):
    P_updated = np.zeros(len(P))
    P_updated[1:] = P[:-1]
    P_updated[0] = P[-1]
    return P_updated
    
def plot_and_get_robot_position_histogram(robot_samples, likelihood):
    # TODO
    # Implementar función
    return histogram