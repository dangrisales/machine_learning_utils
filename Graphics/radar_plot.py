#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:14:25 2023

@author: daniel
"""

import numpy as np
import matplotlib.pyplot as plt

def make_radar_chart_2_models(stats, stats2, attribute_labels, label_model, pathResults, title):
    '''
    Parameters
    ----------
    stats : List
        Lista de los estadisticos del primer modelo.
    stats2 : List
        Lista de los estadisticos del segundo modelo.
    attribute_labels : List
        Lista de las etiquetas de cada estadistico.
    label_model : List
        Lista del nombre de los modelos evaluados.
    pathResults : String
        Dirección de los resultados de la imagen
    title : String
        Nombre del titulo del grafico.

    Returns
    -------
    Guarda la imagen con el grafico de radar en la ruta destino.

    '''
    
    parameters = {'xtick.labelsize': 17, 'ytick.labelsize': 17}
                  
    plt.rcParams.update(parameters)


    labels = np.array(attribute_labels)

    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    stats2 = np.concatenate((stats2,[stats2[0]]))


    angles = np.concatenate((angles,[angles[0]]))

    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(111, polar=True)
    
    ax.plot(angles, stats, 'o-', linewidth=2, c='b', label= label_model[0])
    ax.fill(angles, stats, alpha=0.25, c='b')
    
    ax.plot(angles, stats2, 'o-', linewidth=2, c='orange', label= label_model[1])
    ax.fill(angles, stats2, alpha=0.25, c='orange')
    
    

    ax.set_thetagrids(angles[:-1 ] * 180/np.pi, [])
    #ax.set_ylim((0, max(angles)))
    #ax.set_yticklabels(np.linspace(0, max(angles), len(angles)))
    ax.set_xticklabels(labels)
    ax.set_title(title, fontsize=20)
    ax.grid(True)
    ax.legend(loc=3,fontsize=16)

    fig.savefig(pathResults)

    plt.show()

#Prueba radar
# make_radar_chart_2_models('Test radar', [75, 100, 70, 85, 90], [80, 90, 82, 95, 100],
#                           ['R1', 'R2', 'R3', 'R4', 'R5'],
#                           ['Paciente 1', 'Paciente 2'],
#                           'graphic_radar.pdf', 'Radar prueba')
