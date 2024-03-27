################################################################
#               PRIMER TRABAJO PARA LA MATERIA:                #
#                 APLICACIONES DE LA ANALITICA                 #
#                  MÓDULO DE RECURSOS HUMANOS                  #
#                              POR:                            #
#                                                              #
#                    AURA LUZ MORENO - MOOREA                  #
#                       ALEJANDRA AGUIRRE                      #
#                                                              #
#                    UNIVERSIDAD DE ANTIOQUIA                  #
################################################################

import RH_Grupo1.a_funciones as funciones
import d_modelos as modelos
import pandas as pd
import sqlite3 as sql
import joblib
import openpyxl
import numpy as np
from openpyxl import Workbook


import pandas as pd
import os
from xlsxwriter import Workbook
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

