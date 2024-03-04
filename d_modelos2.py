{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPZbFftLrFif5BKl67a2mFu",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Moorea-AI/ANALITICA3/blob/main/d_modelos2.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "################################################################\n",
        "#               PRIMER TRABAJO PARA LA MATERIA:                #\n",
        "#                 APLICACIONES DE LA ANALITICA                 #\n",
        "#                  MÓDULO DE RECURSOS HUMANOS                  #\n",
        "#                              POR:                            #\n",
        "#                                                              #\n",
        "#                    AURA LUZ MORENO - MOOREA                  #\n",
        "#                       ALEJANDRA AGUIRRE                      #\n",
        "#                                                              #\n",
        "#                    UNIVERSIDAD DE ANTIOQUIA                  #\n",
        "################################################################"
      ],
      "metadata": {
        "id": "wVGYC18KZp3w"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "###importar librerias\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "from itertools import product\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "from plotly.subplots import make_subplots\n",
        "import plotly.express as px\n",
        "import plotly.graph_objects as go\n",
        "import plotly.figure_factory as ff\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.linear_model import LinearRegression\n",
        "import statsmodels.api as sm\n",
        "from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error, r2_score, classification_report\n",
        "import math\n",
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "from sklearn.metrics import accuracy_score\n",
        "import sys\n",
        "import sqlite3 as sql #### para bases de datos sql\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn import linear_model\n",
        "from sklearn import tree\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.ensemble import GradientBoostingRegressor\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from imblearn.combine import SMOTETomek\n",
        "from collections import Counter\n",
        "from sklearn.feature_selection import SelectFromModel\n",
        "from sklearn.linear_model import Lasso\n",
        "from sklearn.feature_selection import RFE"
      ],
      "metadata": {
        "id": "k8_bX0o1gtD_"
      },
      "execution_count": 49,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Importación de datos"
      ],
      "metadata": {
        "id": "3X5U8Yu2PIh5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sys.path.append('/content/drive/MyDrive')"
      ],
      "metadata": {
        "id": "QrX35xCEhPNB"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#conectamos drive\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "c_yd3pReguJl",
        "outputId": "d9b4027c-3371-4bc5-ecef-feed9c1f7bab"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "## conectamos las funciones\n",
        "import a_funciones as funciones"
      ],
      "metadata": {
        "id": "W4ut2VMlP2DA"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "### se genera la URL para conectarse a la base de datos\n",
        "BD = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/databases/BD.csv'"
      ],
      "metadata": {
        "id": "ToTsEkKHYibg"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df=pd.read_csv(BD)"
      ],
      "metadata": {
        "id": "A1TI6tr3YuEv"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df.info() ### mostramos para ver cuales variables son categoricas"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BlXLAS0AaTQf",
        "outputId": "5e3092c3-8dcf-4c0c-ffec-48747ec75b93"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 35280 entries, 0 to 35279\n",
            "Data columns (total 31 columns):\n",
            " #   Column                   Non-Null Count  Dtype  \n",
            "---  ------                   --------------  -----  \n",
            " 0   employeeid               35280 non-null  int64  \n",
            " 1   environmentsatisfaction  35280 non-null  float64\n",
            " 2   jobsatisfaction          35280 non-null  float64\n",
            " 3   worklifebalance          35280 non-null  float64\n",
            " 4   datesurvey               35280 non-null  object \n",
            " 5   jobinvolvement           35280 non-null  int64  \n",
            " 6   performancerating        35280 non-null  int64  \n",
            " 7   surveydate               35280 non-null  object \n",
            " 8   attrition                35280 non-null  object \n",
            " 9   retirementdate           35280 non-null  object \n",
            " 10  retirementtype           35280 non-null  object \n",
            " 11  resignationreason        35280 non-null  object \n",
            " 12  age                      35280 non-null  int64  \n",
            " 13  businesstravel           35280 non-null  object \n",
            " 14  department               35280 non-null  object \n",
            " 15  distancefromhome         35280 non-null  int64  \n",
            " 16  education                35280 non-null  int64  \n",
            " 17  educationfield           35280 non-null  object \n",
            " 18  gender                   35280 non-null  object \n",
            " 19  joblevel                 35280 non-null  int64  \n",
            " 20  jobrole                  35280 non-null  object \n",
            " 21  maritalstatus            35280 non-null  object \n",
            " 22  monthlyincome            35280 non-null  int64  \n",
            " 23  numcompaniesworked       35280 non-null  float64\n",
            " 24  percentsalaryhike        35280 non-null  int64  \n",
            " 25  stockoptionlevel         35280 non-null  int64  \n",
            " 26  totalworkingyears        35280 non-null  float64\n",
            " 27  trainingtimeslastyear    35280 non-null  int64  \n",
            " 28  yearsatcompany           35280 non-null  int64  \n",
            " 29  yearssincelastpromotion  35280 non-null  int64  \n",
            " 30  infodate                 35280 non-null  object \n",
            "dtypes: float64(5), int64(13), object(13)\n",
            "memory usage: 8.3+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "##crreamos una lista con las variables categoricas\n",
        "list_dummies = ['retirementtype', 'resignationreason', 'businesstravel', 'department', 'educationfield', 'gender', 'jobrole', 'maritalstatus']\n"
      ],
      "metadata": {
        "id": "b-SoBBMiY39X"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_dummies=pd.get_dummies(df,columns=list_dummies)\n",
        "df_dummies.info()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "B-jU0SOjaShf",
        "outputId": "3648dd63-1bfa-484c-a4c4-53812408e602"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 35280 entries, 0 to 35279\n",
            "Data columns (total 56 columns):\n",
            " #   Column                             Non-Null Count  Dtype  \n",
            "---  ------                             --------------  -----  \n",
            " 0   employeeid                         35280 non-null  int64  \n",
            " 1   environmentsatisfaction            35280 non-null  float64\n",
            " 2   jobsatisfaction                    35280 non-null  float64\n",
            " 3   worklifebalance                    35280 non-null  float64\n",
            " 4   datesurvey                         35280 non-null  object \n",
            " 5   jobinvolvement                     35280 non-null  int64  \n",
            " 6   performancerating                  35280 non-null  int64  \n",
            " 7   surveydate                         35280 non-null  object \n",
            " 8   attrition                          35280 non-null  object \n",
            " 9   retirementdate                     35280 non-null  object \n",
            " 10  age                                35280 non-null  int64  \n",
            " 11  distancefromhome                   35280 non-null  int64  \n",
            " 12  education                          35280 non-null  int64  \n",
            " 13  joblevel                           35280 non-null  int64  \n",
            " 14  monthlyincome                      35280 non-null  int64  \n",
            " 15  numcompaniesworked                 35280 non-null  float64\n",
            " 16  percentsalaryhike                  35280 non-null  int64  \n",
            " 17  stockoptionlevel                   35280 non-null  int64  \n",
            " 18  totalworkingyears                  35280 non-null  float64\n",
            " 19  trainingtimeslastyear              35280 non-null  int64  \n",
            " 20  yearsatcompany                     35280 non-null  int64  \n",
            " 21  yearssincelastpromotion            35280 non-null  int64  \n",
            " 22  infodate                           35280 non-null  object \n",
            " 23  retirementtype_Fired               35280 non-null  uint8  \n",
            " 24  retirementtype_No                  35280 non-null  uint8  \n",
            " 25  retirementtype_Resignation         35280 non-null  uint8  \n",
            " 26  resignationreason_Activo           35280 non-null  uint8  \n",
            " 27  resignationreason_Others           35280 non-null  uint8  \n",
            " 28  resignationreason_Salary           35280 non-null  uint8  \n",
            " 29  resignationreason_Stress           35280 non-null  uint8  \n",
            " 30  businesstravel_Non-Travel          35280 non-null  uint8  \n",
            " 31  businesstravel_Travel_Frequently   35280 non-null  uint8  \n",
            " 32  businesstravel_Travel_Rarely       35280 non-null  uint8  \n",
            " 33  department_Human Resources         35280 non-null  uint8  \n",
            " 34  department_Research & Development  35280 non-null  uint8  \n",
            " 35  department_Sales                   35280 non-null  uint8  \n",
            " 36  educationfield_Human Resources     35280 non-null  uint8  \n",
            " 37  educationfield_Life Sciences       35280 non-null  uint8  \n",
            " 38  educationfield_Marketing           35280 non-null  uint8  \n",
            " 39  educationfield_Medical             35280 non-null  uint8  \n",
            " 40  educationfield_Other               35280 non-null  uint8  \n",
            " 41  educationfield_Technical Degree    35280 non-null  uint8  \n",
            " 42  gender_Female                      35280 non-null  uint8  \n",
            " 43  gender_Male                        35280 non-null  uint8  \n",
            " 44  jobrole_Healthcare Representative  35280 non-null  uint8  \n",
            " 45  jobrole_Human Resources            35280 non-null  uint8  \n",
            " 46  jobrole_Laboratory Technician      35280 non-null  uint8  \n",
            " 47  jobrole_Manager                    35280 non-null  uint8  \n",
            " 48  jobrole_Manufacturing Director     35280 non-null  uint8  \n",
            " 49  jobrole_Research Director          35280 non-null  uint8  \n",
            " 50  jobrole_Research Scientist         35280 non-null  uint8  \n",
            " 51  jobrole_Sales Executive            35280 non-null  uint8  \n",
            " 52  jobrole_Sales Representative       35280 non-null  uint8  \n",
            " 53  maritalstatus_Divorced             35280 non-null  uint8  \n",
            " 54  maritalstatus_Married              35280 non-null  uint8  \n",
            " 55  maritalstatus_Single               35280 non-null  uint8  \n",
            "dtypes: float64(5), int64(13), object(5), uint8(33)\n",
            "memory usage: 7.3+ MB\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(df_dummies.attrition.unique())\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a2ag09ozRCHm",
        "outputId": "1eda5320-7059-4372-9c85-e431ee79d464"
      },
      "execution_count": 12,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['No' 'Yes']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df_d = df_dummies.drop(['datesurvey', 'surveydate', 'infodate', 'employeeid', 'retirementdate','retirementdate'],  axis=1)"
      ],
      "metadata": {
        "id": "HJSzqqGc9GlY"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ce=[\"distancefromhome\",\"education\",\"joblevel\",\"monthlyincome\",\"percentsalaryhike\",\"stockoptionlevel\",\"yearssincelastpromotion\",\"jobinvolvement\"]\n"
      ],
      "metadata": {
        "id": "kcQmgBJG25jR"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def cambiar_si_no_a_numeros(columna):\n",
        "    return columna.map({'Yes': 1, 'No': 0})\n",
        "\n",
        "# Supongamos que tienes un DataFrame llamado df y quieres cambiar los valores de la columna 'columna_si_no'\n",
        "df_d['attrition'] = cambiar_si_no_a_numeros(df_d['attrition'])"
      ],
      "metadata": {
        "id": "WOhC8amuwOXR"
      },
      "execution_count": 34,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y=df_d[\"attrition\"]\n",
        "x=df_d.drop([\"attrition\"],axis=1)\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5)\n",
        "\n",
        "df_d[ce]=df_d[ce].astype(float)\n",
        "columnasfloat=list(df_d.select_dtypes(\"float64\").columns)\n",
        "pipeline=ColumnTransformer([(\"num\", StandardScaler(),columnasfloat)],remainder='passthrough')\n",
        "\n",
        "X_train_std = pipeline.fit_transform(X_train)\n",
        "X_test_std = pipeline.transform(X_test)"
      ],
      "metadata": {
        "id": "ezcqsigRut76"
      },
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Modelo base"
      ],
      "metadata": {
        "id": "WBsEE66s7UsN"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Entrenamiento del modelo\n",
        "modelo = LogisticRegression(max_iter=1000, solver='sag')\n",
        "# Ajustar el modelo a los datos\n",
        "modelo.fit(X_train_std, y_train)\n",
        "# Desempeño en el entrenamiento\n",
        "y_train_pred = modelo.predict(X_train_std)\n",
        "print('Exactitud en el entrenamiento: %.3f'  %accuracy_score(y_train, y_train_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nSbeVn7f7PqW",
        "outputId": "7687598a-8109-4307-f423-003796eb33ba"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Exactitud en el entrenamiento: 1.000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = modelo.predict(X_test_std)\n",
        "# Exactitud en el conjunto de validación\n",
        "print('Exactitud en la validacion: %.3f'  %accuracy_score(y_test, y_pred))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yOOWCzz9xvK6",
        "outputId": "3be2afe0-257a-4b68-8da5-253d240cbeda"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Exactitud en la validacion: 1.000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "matriz= confusion_matrix(y_test, y_pred)\n",
        "matriz_display = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=['No renuncia', 'renuncia'])\n",
        "matriz_display.plot()\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "m9spmH1N3VU_",
        "outputId": "1491a65f-8cf8-4682-93e9-9862a3027739"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAlcAAAGwCAYAAACEkkAjAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjcuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/bCgiHAAAACXBIWXMAAA9hAAAPYQGoP6dpAABUvklEQVR4nO3deVgVZf8/8PcBhAPIOQgqSCJg7oqiWIr7QmIuidljGJkL6S+TFM0lHxX3DdMUM0nNRL/6mGbyuIUS5E4qKK6IG+4CGgKCAYdz5vcHD4NHUDnOICDv13XNdXVm7rnnM4aeD5/7nnsUgiAIICIiIiJZGJV1AERERERvEiZXRERERDJickVEREQkIyZXRERERDJickVEREQkIyZXRERERDJickVEREQkI5OyDoBeD51Oh3v37sHKygoKhaKswyEiIgMJgoDHjx/DwcEBRkalUxvJzs5Gbm6uLH2ZmppCqVTK0ldFw+Sqkrh37x4cHR3LOgwiIpLo9u3bqF27tuz9Zmdnw8WpKpJStLL0Z29vj8TExEqZYDG5qiSsrKwAADdPOUNVlaPB9Gbq38C1rEMgKjV50OAI9or/nsstNzcXSSla3Ix1hspK2vdExmMdnNxvIDc3l8kVvbkKhgJVVY0k/6UhKq9MFFXKOgSi0vO/l9WV9tSOqlYKVLWSdg0dKvf0EyZXREREJNIKOmglvnVYK+jkCaaCYnJFREREIh0E6CAtu5J6fkXH8SEiIiIiGbFyRURERCIddJA6qCe9h4qNyRURERGJtIIArSBtWE/q+RUdhwWJiIiIZMTKFREREYk4oV06JldEREQk0kGAlsmVJBwWJCIiIpIRK1dEREQk4rCgdEyuiIiISMSnBaXjsCARERGRjFi5IiIiIpHuf5vUPiozJldEREQk0srwtKDU8ys6DgsSERGRSCvIsxni0KFD6Nu3LxwcHKBQKBAWFvbctl988QUUCgWWLVumtz81NRW+vr5QqVSwtraGn58fMjMz9dqcPXsWHTt2hFKphKOjI4KCgor0v23bNjRq1AhKpRKurq7Yu3evYTcDJldERERUxrKystCiRQusXLnyhe127NiBv/76Cw4ODkWO+fr64sKFC4iIiMDu3btx6NAhjBw5UjyekZGBHj16wMnJCbGxsVi8eDFmzpyJ1atXi22OHTuGQYMGwc/PD6dPn4a3tze8vb1x/vx5g+6Hw4JEREQkknPOVUZGht5+MzMzmJmZFWn//vvv4/33339hn3fv3sVXX32Fffv2oXfv3nrH4uPjER4ejpMnT6J169YAgBUrVqBXr1749ttv4eDggE2bNiE3Nxfr1q2DqakpmjZtiri4OCxdulRMwpYvX46ePXti4sSJAIA5c+YgIiIC33//PUJCQkp8/6xcERERkUgHBbQSNx0UAABHR0eo1WpxW7BgwavFpNNh8ODBmDhxIpo2bVrkeHR0NKytrcXECgA8PT1hZGSE48ePi206deoEU1NTsY2XlxcSEhLw6NEjsY2np6de315eXoiOjjYoXlauiIiIqFTcvn0bKpVK/Fxc1aokFi1aBBMTE4wZM6bY40lJSahZs6bePhMTE9jY2CApKUls4+LiotfGzs5OPFatWjUkJSWJ+55uU9BHSTG5IiIiIpFOyN+k9gEAKpVKL7l6FbGxsVi+fDlOnToFhUIhLbDXhMOCREREJJI6JFiwyeXw4cNISUlBnTp1YGJiAhMTE9y8eRNff/01nJ2dAQD29vZISUnROy8vLw+pqamwt7cX2yQnJ+u1Kfj8sjYFx0uKyRURERGVW4MHD8bZs2cRFxcnbg4ODpg4cSL27dsHAPDw8EBaWhpiY2PF86KioqDT6dCmTRuxzaFDh6DRaMQ2ERERaNiwIapVqya2iYyM1Lt+REQEPDw8DIqZw4JEREQkkqPyZOj5mZmZuHr1qvg5MTERcXFxsLGxQZ06dWBra6vXvkqVKrC3t0fDhg0BAI0bN0bPnj0xYsQIhISEQKPRwN/fHz4+PuKyDZ988glmzZoFPz8/TJ48GefPn8fy5cvx3Xffif2OHTsWnTt3xpIlS9C7d29s2bIFMTExess1lAQrV0RERCTSCQpZNkPExMSgZcuWaNmyJQBg/PjxaNmyJQIDA0vcx6ZNm9CoUSN0794dvXr1QocOHfSSIrVajf379yMxMRHu7u74+uuvERgYqLcWVrt27bB582asXr0aLVq0wK+//oqwsDA0a9bMoPtRCEIlf3V1JZGRkQG1Wo1Hl+tCZcWcmt5MXg5uZR0CUanJEzQ4gP8iPT1d8iTx4hR8Txw574CqEr8nMh/r0KHZvVKLtbzjsCARERGJymJY8E3D5IqIiIhEWhhBK3HWkFamWCoqJldEREQkEl5hzlRxfVRmnHxDREREJCNWroiIiEjEOVfSMbkiIiIikVYwglaQOOeqkq9DwGFBIiIiIhmxckVEREQiHRTQSay96FC5S1dMroiIiEjEOVfScViQiIiISEasXBEREZFIngntHBYkIiIiAlAw50rasJ7U8ys6DgsSERERyYiVKyIiIhLpZHi3IJ8WJCIiIvofzrmSjskVERERiXQw4jpXEnHOFREREZGMWLkiIiIikVZQQCtIXERU4vkVHZMrIiIiEmllmNCu5bAgEREREcmFlSsiIiIS6QQj6CQ+Lajj04JERERE+TgsKB2HBYmIiIhkxMoVERERiXSQ/rSfTp5QKiwmV0RERCSSZxHRyj0wVrnvnoiIiEhmrFwRERGRSJ53C1bu2g2TKyIiIhLpoIAOUudccYV2IiIiIgCsXMmhct89ERERkcxYuSIiIiKRPIuIVu7aDZMrIiIiEukEBXRS17mSeH5FV7lTSyIiIiKZsXJFREREIp0Mw4KVfRFRJldEREQk0glG0El82k/q+RVd5b57IiIiIpmxckVEREQiLRTQSlwEVOr5FR2TKyIiIhJxWFC6yn33REREVOYOHTqEvn37wsHBAQqFAmFhYeIxjUaDyZMnw9XVFZaWlnBwcMBnn32Ge/fu6fWRmpoKX19fqFQqWFtbw8/PD5mZmXptzp49i44dO0KpVMLR0RFBQUFFYtm2bRsaNWoEpVIJV1dX7N271+D7YXJFREREIi0KhwZffTNMVlYWWrRogZUrVxY59uTJE5w6dQrTp0/HqVOn8NtvvyEhIQEffPCBXjtfX19cuHABERER2L17Nw4dOoSRI0eKxzMyMtCjRw84OTkhNjYWixcvxsyZM7F69WqxzbFjxzBo0CD4+fnh9OnT8Pb2hre3N86fP2/Q/SgEQRAM/DOgCigjIwNqtRqPLteFyoo5Nb2ZvBzcyjoEolKTJ2hwAP9Feno6VCqV7P0XfE9M+6sHlFWrSOorO1ODuW33v1KsCoUCO3bsgLe393PbnDx5Eu+++y5u3ryJOnXqID4+Hk2aNMHJkyfRunVrAEB4eDh69eqFO3fuwMHBAatWrcLUqVORlJQEU1NTAMA333yDsLAwXLp0CQDw8ccfIysrC7t37xav1bZtW7i5uSEkJKTE98BvWSIiIhIVvLhZ6gbkJ2xPbzk5ObLEmJ6eDoVCAWtrawBAdHQ0rK2txcQKADw9PWFkZITjx4+LbTp16iQmVgDg5eWFhIQEPHr0SGzj6empdy0vLy9ER0cbFB+TKyIiIioVjo6OUKvV4rZgwQLJfWZnZ2Py5MkYNGiQWBVLSkpCzZo19dqZmJjAxsYGSUlJYhs7Ozu9NgWfX9am4HhJ8WlBIiIiEglQQCdxKQXhf+ffvn1bb1jQzMxMUr8ajQYDBw6EIAhYtWqVpL5KE5MrIiIiEj09rCelDwBQqVSyzQ8rSKxu3ryJqKgovX7t7e2RkpKi1z4vLw+pqamwt7cX2yQnJ+u1Kfj8sjYFx0uKw4JERERUrhUkVleuXMEff/wBW1tbveMeHh5IS0tDbGysuC8qKgo6nQ5t2rQR2xw6dAgajUZsExERgYYNG6JatWpim8jISL2+IyIi4OHhYVC8TK6IiIhIpBMUsmyGyMzMRFxcHOLi4gAAiYmJiIuLw61bt6DRaPDRRx8hJiYGmzZtglarRVJSEpKSkpCbmwsAaNy4MXr27IkRI0bgxIkTOHr0KPz9/eHj4wMHBwcAwCeffAJTU1P4+fnhwoUL+OWXX7B8+XKMHz9ejGPs2LEIDw/HkiVLcOnSJcycORMxMTHw9/c36H44LEhEREQiLYyglVh7MfT8mJgYdO3aVfxckPAMGTIEM2fOxM6dOwEAbm5ueuf9+eef6NKlCwBg06ZN8Pf3R/fu3WFkZIQBAwYgODhYbKtWq7F//36MHj0a7u7uqF69OgIDA/XWwmrXrh02b96MadOm4d///jfq16+PsLAwNGvWzKD74TpXlQTXuaLKgOtc0Zvsda1zFXD0A5hJXOcqJ1ODZe13llqs5R0rV0RERCR6lWG94vqozJhcERERkUgHI+gkDgtKPb+iq9x3T0RERCQzVq6IiIhIpBUU0Eoc1pN6fkXH5IqIiIhEnHMlHZMrIiIiEgmCEXQSV2gXJJ5f0VXuuyciIiKSGStXREREJNJCAa3EFzdLPb+iY3JFREREIp0gfc6UrpIvT85hQSIiIiIZsXJF9D/n/rLEth9q4so5C6QmV8GMnxLR7v30Ytsun1wbezdWx/+bdRcfjngg7r9zzQxr5jjg4klL5GkUcGn8Dz6blAS39plim5Q7VbBiSm2cOWoFpaUW7/3rEYb/+x6M//e38fxxS/w0rxZuX1Mi5x8j1HwrF70H/40PRz54NgyiMtN36EN8NCoFNjXycP2iOX6Y9hYS4izKOiySgU6GCe1Sz6/oKvfdl2NDhw6Ft7d3WYdRqWQ/MULdpv/Af/6dF7Y7+rsal2ItYWufW+RY4BAX6LTAom1X8X14Auo2+QeBn7kgNSU/c9Jqgemf1YUm1wjf7byCictvIWKrDUIX1xL7UFro8MGwh/j2t6tYc/ASPglIxvpF9tj7f7by3jDRK+r8wSOMnHEPm5baY7RXA1y/qMS8zdehttWUdWgkAx0UsmyVWZkmV0OHDoVCocDChQv19oeFhUGhqNz/Y5YvX47169eXdRiVyjvdHmPo5CS0f061CgAe3q+CH6a9hckrb8Lkmbpv+t/GuHtdiYH+KajbJBtv1c3F8Kn3kfOPMW5cUgIATh20wq3LSkz+/ibebvYP3un2GJ9Nuo9d66tDk5v/M1/P9R907Z8G54bZsHfMRfcBj9C6y2OcP25ZavdOZIgPRz5E+GYb7P/FBreuKBE8uTZy/lHAa1BqWYdGVC6UeeVKqVRi0aJFePToUaleJze3aJWhPFOr1bC2ti7rMOgpOh0QNKYOPhqVAueG2UWOq2y0qP12Nv7YZoPsJ0bQ5gF7NtrCuroG9Zv/AwC4GGMJ50bZqFYjTzyvdZfHePLYGDcTlMVe9+o5c1yMsYRr28xijxO9TiZVdKjf/AlOHbYS9wmCAqcPW6GJ+5MyjIzkUrBCu9StMivz5MrT0xP29vZYsGDBC9tt374dTZs2hZmZGZydnbFkyZIXtp85cybc3Nywdu1auLi4QKnM/+JKS0vD559/jho1akClUqFbt244c+ZMkfM2btwIZ2dnqNVq+Pj44PHjx2IbZ2dnLFu2TO96bm5umDlzpvhZoVBg7dq16N+/PywsLFC/fn3s3LlT75wLFy6gT58+UKlUsLKyQseOHXHt2jUARYcFw8PD0aFDB1hbW8PW1hZ9+vQR29LrsXVlTRgbC/D2e1jscYUCWPjLNVw7bw7v+q7o49ICv62uiXmbrsPKWgsAePTABNVq6A+dWFfXiMee5uveBH2cm+Or9xug79CHeN+XVQEqeyobLYxNgLRnfl4fPTTR+6WBKq6COVdSt8qszO/e2NgY8+fPx4oVK3DnTvFzXWJjYzFw4ED4+Pjg3LlzmDlzJqZPn/7SYbOrV69i+/bt+O233xAXFwcA+Ne//oWUlBT8/vvviI2NRatWrdC9e3ekphZ+cV27dg1hYWHYvXs3du/ejYMHDxYZuiyJWbNmYeDAgTh79ix69eoFX19f8Tp3795Fp06dYGZmhqioKMTGxmL48OHIyyv+H6esrCyMHz8eMTExiIyMhJGREfr37w+dTlds+5ycHGRkZOht9OqunDVH2NoamLDsFp43Yi0IwPf/rg3r6nlYsuMqgvdcRrue6Zgx1AV/Jxv+7MiSHVex4vfL+GrRbexYWwN/7rCWdhNERPRalIunBfv37w83NzfMmDEDP/30U5HjS5cuRffu3TF9+nQAQIMGDXDx4kUsXrwYQ4cOfW6/ubm52LBhA2rUqAEAOHLkCE6cOIGUlBSYmZkBAL799luEhYXh119/xciRIwEAOp0O69evh5VVftl78ODBiIyMxLx58wy6r6FDh2LQoEEAgPnz5yM4OBgnTpxAz549sXLlSqjVamzZsgVVqlQR7+t5BgwYoPd53bp1qFGjBi5evIhmzZoVab9gwQLMmjXLoHjp+c4dr4q0hyb49J2m4j6dVoE1sxwQtqYGNpy4iLgjVXHiDxV+jT8HS6v8pLd+8zs4dagx/thqg4+/SkG1GnlIOK0/dyrtYf7//2d/67evkz+U7dI4G2kPquD/ltija/+0UrxLopfLSDWGNg+wfubntVr1vCLVV6qYdJDh3YKc0F4+LFq0CKGhoYiPjy9yLD4+Hu3bt9fb1759e1y5cgVarfa5fTo5OYmJFQCcOXMGmZmZsLW1RdWqVcUtMTFRb4jN2dlZTKwAoFatWkhJSTH4npo3by7+t6WlJVQqldhPXFwcOnbsKCZWL3PlyhUMGjQIdevWhUqlgrOzMwDg1q1bxbafMmUK0tPTxe327dsGx0+FPAekIiQyAasiCjdb+1x8NCoF8zbn/+zk/JP/18nomb9VRgpBXFCvSess3LikRNrDwi+hU4esYGGlRZ0GRedxFdDpAE1uufnrSpVYnsYIV85aoGWHwqkSCoUAtw6ZuBjLpRjeBIIMTwoKlTy5Kje/ZnTq1AleXl6YMmXKC6tRhrC01K8QZGZmolatWjhw4ECRtk9PHn824VEoFHrDb0ZGRhAE/eVnNZqijyC/qB9zc/MS3UOBvn37wsnJCWvWrIGDgwN0Oh2aNWv23In6ZmZmYnWOSuafLCPcSyz8M0u6bYpr581hZZ2HmrU1UNnoJ/ImJkC1mnlwrJcDAGjsnoWqai0Wj60D33FJMFMK+H2TLZJum+Ld7vnDsq06P0adBtkI+qoO/Kbdw6MHVbB+kT36Dn0IU7P8n6mdP1dHzbdy4VgvP9k691dVbA+piX5+XOeKyoffVlfHhGW3cfmMBRJOW6D/iAdQWuiwf4tNWYdGMtAJMlSuKvmE9nKTXAHAwoUL4ebmhoYNG+rtb9y4MY4ePaq37+jRo2jQoAGMjY1L3H+rVq2QlJQEExMTsfLzKmrUqIH79++LnzMyMpCYmGhQH82bN0doaCg0Gs1Lq1d///03EhISsGbNGnTs2BFA/hAnyevyGQtM+qie+PnHmW8BAN4bmIoJy4qvED5NbavFvM3XsH5hLUweWA9ajQJODbMx8+dEvN00P1EyNgZmb7iOFd84YlzfBlBa6OD5r1QMmVj48yTogHULaiHplimMTQAHpxwMn3oPvQf/LfMdE72agzurQW2rxWcTk1CtRh6uXzDHVF8XcYibqLIrV8mVq6srfH19ERwcrLf/66+/xjvvvIM5c+bg448/RnR0NL7//nv88MMPBvXv6ekJDw8PeHt7IygoCA0aNMC9e/ewZ88e9O/fH61bty5RP926dcP69evRt29fWFtbIzAw0KAkDwD8/f2xYsUK+Pj4YMqUKVCr1fjrr7/w7rvvFkkuq1WrBltbW6xevRq1atXCrVu38M033xh0PXq5Fu0yse9eXInbbzhxsci+Bi3+wfz/XH/heXa1NZj7f89v08/vIfo954lEovJi58/VsfPn6mUdBpUCrtAuXbm7+9mzZxd5Aq5Vq1bYunUrtmzZgmbNmiEwMBCzZ882ePhQoVBg79696NSpE4YNG4YGDRrAx8cHN2/ehJ2dXYn7mTJlCjp37ow+ffqgd+/e8Pb2xttvv21QLLa2toiKikJmZiY6d+4Md3d3rFmzptgqlpGREbZs2YLY2Fg0a9YM48aNw+LFiw26HhERUUkUDAtK3SozhfDs5CF6I2VkZECtVuPR5bpQWZW7nJpIFl4ObmUdAlGpyRM0OID/Ij09HSqVSvb+C74n+u0fjiqWppL60mTl4r891pVarOVduRoWJCIiorIlx7sBK/tSDEyuiIiISMSnBaXj+BARERGRjFi5IiIiIhErV9IxuSIiIiIRkyvpOCxIREREJCNWroiIiEjEypV0TK6IiIhIJED6UgqVfQFNJldEREQkYuVKOs65IiIiIpIRK1dEREQkYuVKOiZXREREJGJyJR2HBYmIiIhkxMoVERERiVi5ko7JFREREYkEQQFBYnIk9fyKjsOCRERERDJickVEREQiHRSybIY4dOgQ+vbtCwcHBygUCoSFhekdFwQBgYGBqFWrFszNzeHp6YkrV67otUlNTYWvry9UKhWsra3h5+eHzMxMvTZnz55Fx44doVQq4ejoiKCgoCKxbNu2DY0aNYJSqYSrqyv27t1r0L0ATK6IiIjoKQVzrqRuhsjKykKLFi2wcuXKYo8HBQUhODgYISEhOH78OCwtLeHl5YXs7Gyxja+vLy5cuICIiAjs3r0bhw4dwsiRI8XjGRkZ6NGjB5ycnBAbG4vFixdj5syZWL16tdjm2LFjGDRoEPz8/HD69Gl4e3vD29sb58+fN+h+FIIgVPZV6iuFjIwMqNVqPLpcFyor5tT0ZvJycCvrEIhKTZ6gwQH8F+np6VCpVLL3X/A90SZsDEwszST1lZeVg+Pewa8Uq0KhwI4dO+Dt7Q0gv2rl4OCAr7/+GhMmTAAApKenw87ODuvXr4ePjw/i4+PRpEkTnDx5Eq1btwYAhIeHo1evXrhz5w4cHBywatUqTJ06FUlJSTA1NQUAfPPNNwgLC8OlS5cAAB9//DGysrKwe/duMZ62bdvCzc0NISEhJb4HfssSERGRqGBCu9QNyE/Ynt5ycnIMjicxMRFJSUnw9PQU96nVarRp0wbR0dEAgOjoaFhbW4uJFQB4enrCyMgIx48fF9t06tRJTKwAwMvLCwkJCXj06JHY5unrFLQpuE5JMbkiIiIikZzDgo6OjlCr1eK2YMECg+NJSkoCANjZ2entt7OzE48lJSWhZs2aesdNTExgY2Oj16a4Pp6+xvPaFBwvKS7FQERERCI5l2K4ffu23rCgmZm04caKgpUrIiIiKhUqlUpve5Xkyt7eHgCQnJystz85OVk8Zm9vj5SUFL3jeXl5SE1N1WtTXB9PX+N5bQqOlxSTKyIiIhIJMgwJyrmIqIuLC+zt7REZGSnuy8jIwPHjx+Hh4QEA8PDwQFpaGmJjY8U2UVFR0Ol0aNOmjdjm0KFD0Gg0YpuIiAg0bNgQ1apVE9s8fZ2CNgXXKSkmV0RERCQSAAiCxM3Aa2ZmZiIuLg5xcXEA8iexx8XF4datW1AoFAgICMDcuXOxc+dOnDt3Dp999hkcHBzEJwobN26Mnj17YsSIEThx4gSOHj0Kf39/+Pj4wMHBAQDwySefwNTUFH5+frhw4QJ++eUXLF++HOPHjxfjGDt2LMLDw7FkyRJcunQJM2fORExMDPz9/Q26H865IiIiojIVExODrl27ip8LEp4hQ4Zg/fr1mDRpErKysjBy5EikpaWhQ4cOCA8Ph1KpFM/ZtGkT/P390b17dxgZGWHAgAEIDg4Wj6vVauzfvx+jR4+Gu7s7qlevjsDAQL21sNq1a4fNmzdj2rRp+Pe//4369esjLCwMzZo1M+h+uM5VJcF1rqgy4DpX9CZ7Xetctfj1axhbSJt4rn2SgzMfLSm1WMs7Vq6IiIhIxBc3S8cSBhEREZGMWLkiIiIikU5QQCGx8mTouwXfNEyuiIiISFTwxJ/UPiozDgsSERERyYiVKyIiIhJxQrt0TK6IiIhIxORKOiZXREREJOKEduk454qIiIhIRqxcERERkYhPC0rH5IqIiIhE+cmV1DlXMgVTQXFYkIiIiEhGrFwRERGRiE8LSsfkioiIiETC/zapfVRmHBYkIiIikhErV0RERCTisKB0TK6IiIioEMcFJWNyRURERIVkqFyhkleuOOeKiIiISEasXBEREZGIK7RLx+SKiIiIRJzQLh2HBYmIiIhkxMoVERERFRIU0iekV/LKFZMrIiIiEnHOlXQcFiQiIiKSEStXREREVIiLiErG5IqIiIhEfFpQuhIlVzt37ixxhx988MErB0NERERU0ZUoufL29i5RZwqFAlqtVko8REREVNYq+bCeVCVKrnQ6XWnHQUREROUAhwWlk/S0YHZ2tlxxEBERUXkgyLRVYgYnV1qtFnPmzMFbb72FqlWr4vr16wCA6dOn46effpI9QCIiIqKKxODkat68eVi/fj2CgoJgamoq7m/WrBnWrl0ra3BERET0uilk2iovg5OrDRs2YPXq1fD19YWxsbG4v0WLFrh06ZKswREREdFrxmFByQxOru7evYt69eoV2a/T6aDRaGQJioiIiKiiMji5atKkCQ4fPlxk/6+//oqWLVvKEhQRERGVEVauJDN4hfbAwEAMGTIEd+/ehU6nw2+//YaEhARs2LABu3fvLo0YiYiI6HURFPmb1D4qMYMrV/369cOuXbvwxx9/wNLSEoGBgYiPj8euXbvw3nvvlUaMRERERBXGK71bsGPHjoiIiJA7FiIiIipjgpC/Se2jMnvlRURjYmKwceNGbNy4EbGxsXLGRERERGWlDOZcabVaTJ8+HS4uLjA3N8fbb7+NOXPmQHgqSxMEAYGBgahVqxbMzc3h6emJK1eu6PWTmpoKX19fqFQqWFtbw8/PD5mZmXptzp49i44dO0KpVMLR0RFBQUGGBVsCBleu7ty5g0GDBuHo0aOwtrYGAKSlpaFdu3bYsmULateuLXeMRERE9AZbtGgRVq1ahdDQUDRt2hQxMTEYNmwY1Go1xowZAwAICgpCcHAwQkND4eLigunTp8PLywsXL16EUqkEAPj6+uL+/fuIiIiARqPBsGHDMHLkSGzevBkAkJGRgR49esDT0xMhISE4d+4chg8fDmtra4wcOVK2+zG4cvX5559Do9EgPj4eqampSE1NRXx8PHQ6HT7//HPZAiMiIqIyUDChXepmgGPHjqFfv37o3bs3nJ2d8dFHH6FHjx44ceJEfkiCgGXLlmHatGno168fmjdvjg0bNuDevXsICwsDAMTHxyM8PBxr165FmzZt0KFDB6xYsQJbtmzBvXv3AACbNm1Cbm4u1q1bh6ZNm8LHxwdjxozB0qVLZf0jNDi5OnjwIFatWoWGDRuK+xo2bIgVK1bg0KFDsgZHREREr5dCkGcD8itFT285OTnFXrNdu3aIjIzE5cuXAQBnzpzBkSNH8P777wMAEhMTkZSUBE9PT/EctVqNNm3aIDo6GgAQHR0Na2trtG7dWmzj6ekJIyMjHD9+XGzTqVMnvTfMeHl5ISEhAY8ePZLtz9Dg5MrR0bHYxUK1Wi0cHBxkCYqIiIjKiIxzrhwdHaFWq8VtwYIFxV7ym2++gY+PDxo1aoQqVaqgZcuWCAgIgK+vLwAgKSkJAGBnZ6d3np2dnXgsKSkJNWvW1DtuYmICGxsbvTbF9fH0NeRg8JyrxYsX46uvvsLKlSvF7DAmJgZjx47Ft99+K1tgREREVLHdvn0bKpVK/GxmZlZsu61bt2LTpk3YvHkzmjZtiri4OAQEBMDBwQFDhgx5XeHKpkTJVbVq1aBQFI6fZmVloU2bNjAxyT89Ly8PJiYmGD58OLy9vUslUCIiInoNZFxEVKVS6SVXzzNx4kSxegUArq6uuHnzJhYsWIAhQ4bA3t4eAJCcnIxatWqJ5yUnJ8PNzQ0AYG9vj5SUFL1+8/LykJqaKp5vb2+P5ORkvTYFnwvayKFEydWyZctkuyARERGVY3K8vsbA8588eQIjI/2ZSsbGxtDpdAAAFxcX2NvbIzIyUkymMjIycPz4cYwaNQoA4OHhgbS0NMTGxsLd3R0AEBUVBZ1OhzZt2ohtpk6dCo1GgypVqgAAIiIi0LBhQ1SrVu1V77aIEiVXFbEkR0RERBVD3759MW/ePNSpUwdNmzbF6dOnsXTpUgwfPhwAoFAoEBAQgLlz56J+/friUgwODg7iiFnjxo3Rs2dPjBgxAiEhIdBoNPD394ePj484J/yTTz7BrFmz4Ofnh8mTJ+P8+fNYvnw5vvvuO1nv55VWaC+QnZ2N3NxcvX0lKf8RERFROVUGlasVK1Zg+vTp+PLLL5GSkgIHBwf8v//3/xAYGCi2mTRpErKysjBy5EikpaWhQ4cOCA8PF9e4AvKXWvD390f37t1hZGSEAQMGIDg4WDyuVquxf/9+jB49Gu7u7qhevToCAwNlXeMKABSCYNgi9VlZWZg8eTK2bt2Kv//+u8hxrVYrW3Akn4yMDKjVajy6XBcqq1demJ+oXPNycCvrEIhKTZ6gwQH8F+np6aVSyCj4nnD8dg6MzJUvP+EFdP9k4/aE6aUWa3ln8LfspEmTEBUVhVWrVsHMzAxr167FrFmz4ODggA0bNpRGjEREREQVhsHDgrt27cKGDRvQpUsXDBs2DB07dkS9evXg5OSETZs2iWtSEBERUQUk49OClZXBlavU1FTUrVsXQP78qtTUVABAhw4duEI7ERFRBSfnCu2VlcHJVd26dZGYmAgAaNSoEbZu3Qogv6JV8CJnIiIiosrK4ORq2LBhOHPmDID85epXrlwJpVKJcePGYeLEibIHSERERK+RjK+/qawMnnM1btw48b89PT1x6dIlxMbGol69emjevLmswRERERFVNJLWuQIAJycnODk5yRELERERlTEFpM+ZqtzT2UuYXD29ANfLjBkz5pWDISIiIqroSpRclXRZeIVCweSqnOvfwBUmiiplHQZRqdB2aVXWIRCVGm1eNnD4v6V/IS7FIFmJkquCpwOJiIjoDVcGr7950/A9KEREREQykjyhnYiIiN4grFxJxuSKiIiIRHKssM4V2omIiIhINqxcERERUSEOC0r2SpWrw4cP49NPP4WHhwfu3r0LANi4cSOOHDkia3BERET0mvH1N5IZnFxt374dXl5eMDc3x+nTp5GTkwMASE9Px/z582UPkIiIiKgiMTi5mjt3LkJCQrBmzRpUqVK4GGX79u1x6tQpWYMjIiKi16tgQrvUrTIzeM5VQkICOnXqVGS/Wq1GWlqaHDERERFRWeEK7ZIZXLmyt7fH1atXi+w/cuQI6tatK0tQREREVEY450oyg5OrESNGYOzYsTh+/DgUCgXu3buHTZs2YcKECRg1alRpxEhERERUYRg8LPjNN99Ap9Ohe/fuePLkCTp16gQzMzNMmDABX331VWnESERERK8JFxGVzuDkSqFQYOrUqZg4cSKuXr2KzMxMNGnSBFWrVi2N+IiIiOh14jpXkr3yIqKmpqZo0qSJnLEQERERVXgGJ1ddu3aFQvH8pwCioqIkBURERERlSI6lFFi5Moybm5veZ41Gg7i4OJw/fx5DhgyRKy4iIiIqCxwWlMzg5Oq7774rdv/MmTORmZkpOSAiIiKiiuyV3i1YnE8//RTr1q2TqzsiIiIqC1znSrJXntD+rOjoaCiVSrm6IyIiojLApRikMzi5+vDDD/U+C4KA+/fvIyYmBtOnT5ctMCIiIqKKyODkSq1W6302MjJCw4YNMXv2bPTo0UO2wIiIiIgqIoOSK61Wi2HDhsHV1RXVqlUrrZiIiIiorPBpQckMmtBubGyMHj16IC0trZTCISIiorJUMOdK6laZGfy0YLNmzXD9+vXSiIWIiIiowjM4uZo7dy4mTJiA3bt34/79+8jIyNDbiIiIqILjMgySlHjO1ezZs/H111+jV69eAIAPPvhA7zU4giBAoVBAq9XKHyURERG9HpxzJVmJk6tZs2bhiy++wJ9//lma8RARERFVaCVOrgQhPw3t3LlzqQVDREREZYuLiEpn0FIMTw8DEhER0RuIw4KSGTShvUGDBrCxsXnhRkRERGSou3fv4tNPP4WtrS3Mzc3h6uqKmJgY8bggCAgMDEStWrVgbm4OT09PXLlyRa+P1NRU+Pr6QqVSwdraGn5+fsjMzNRrc/bsWXTs2BFKpRKOjo4ICgqS/V4MqlzNmjWryArtRERE9OYoi2HBR48eoX379ujatSt+//131KhRA1euXNFbsDwoKAjBwcEIDQ2Fi4sLpk+fDi8vL1y8eFF8t7Gvry/u37+PiIgIaDQaDBs2DCNHjsTmzZsBABkZGejRowc8PT0REhKCc+fOYfjw4bC2tsbIkSOl3fRTDEqufHx8ULNmTdkuTkREROVMGQwLLlq0CI6Ojvj555/FfS4uLoXdCQKWLVuGadOmoV+/fgCADRs2wM7ODmFhYfDx8UF8fDzCw8Nx8uRJtG7dGgCwYsUK9OrVC99++y0cHBywadMm5ObmYt26dTA1NUXTpk0RFxeHpUuXyppclXhYkPOtiIiIyBDProWZk5NTbLudO3eidevW+Ne//oWaNWuiZcuWWLNmjXg8MTERSUlJ8PT0FPep1Wq0adMG0dHRAIDo6GhYW1uLiRUAeHp6wsjICMePHxfbdOrUCaampmIbLy8vJCQk4NGjR7Ldd4mTq4KnBYmIiOgNJnUB0acqX46OjlCr1eK2YMGCYi95/fp1rFq1CvXr18e+ffswatQojBkzBqGhoQCApKQkAICdnZ3eeXZ2duKxpKSkIqNrJiYmsLGx0WtTXB9PX0MOJR4W1Ol0sl2UiIiIyic551zdvn0bKpVK3G9mZlZse51Oh9atW2P+/PkAgJYtW+L8+fMICQnBkCFDpAVTBgx+/Q0RERG9wWSsXKlUKr3teclVrVq10KRJE719jRs3xq1btwAA9vb2AIDk5GS9NsnJyeIxe3t7pKSk6B3Py8tDamqqXpvi+nj6GnJgckVERERlqn379khISNDbd/nyZTg5OQHIn9xub2+PyMhI8XhGRgaOHz8ODw8PAICHhwfS0tIQGxsrtomKioJOp0ObNm3ENocOHYJGoxHbREREoGHDhnpPJkrF5IqIiIgKyVi5Kqlx48bhr7/+wvz583H16lVs3rwZq1evxujRowHkP1QXEBCAuXPnYufOnTh37hw+++wzODg4wNvbG0B+patnz54YMWIETpw4gaNHj8Lf3x8+Pj5wcHAAAHzyyScwNTWFn58fLly4gF9++QXLly/H+PHjJfyBFWXQUgxERET0ZiuLda7eeecd7NixA1OmTMHs2bPh4uKCZcuWwdfXV2wzadIkZGVlYeTIkUhLS0OHDh0QHh4urnEFAJs2bYK/vz+6d+8OIyMjDBgwAMHBweJxtVqN/fv3Y/To0XB3d0f16tURGBgo6zIMAKAQ+BhgpZCRkQG1Wo0u6AcTRZWyDoeoVGi7tCrrEIhKTV5eNg4fno309HS9SeJyKfieaDRmPozNlC8/4QW0Odm4FPzvUou1vGPlioiIiArx3YKSMbkiIiIiUVkMC75pOKGdiIiISEasXBEREVEhDgtKxuSKiIiICjG5kozDgkREREQyYuWKiIiIRIr/bVL7qMyYXBEREVEhDgtKxuSKiIiIRFyKQTrOuSIiIiKSEStXREREVIjDgpIxuSIiIiJ9lTw5korDgkREREQyYuWKiIiIRJzQLh2TKyIiIirEOVeScViQiIiISEasXBEREZGIw4LSMbkiIiKiQhwWlIzDgkREREQyYuWKiIiIRBwWlI7JFRERERXisKBkTK6IiIioEJMryTjnioiIiEhGrFwRERGRiHOupGNyRURERIU4LCgZhwWJiIiIZMTKFREREYkUggCFIK30JPX8io7JFRERERXisKBkHBYkIiIikhErV0RERCTi04LSMbkiIiKiQhwWlIzDgkREREQyYuWKiIiIRBwWlI7JFRERERXisKBkTK6IiIhIxMqVdJxzRURERCQjVq6IiIioEIcFJWNyRURERHoq+7CeVBwWJCIiIpIRkysiIiIqJAjybK9o4cKFUCgUCAgIEPdlZ2dj9OjRsLW1RdWqVTFgwAAkJyfrnXfr1i307t0bFhYWqFmzJiZOnIi8vDy9NgcOHECrVq1gZmaGevXqYf369a8c54swuSIiIiJRwdOCUrdXcfLkSfz4449o3ry53v5x48Zh165d2LZtGw4ePIh79+7hww8/FI9rtVr07t0bubm5OHbsGEJDQ7F+/XoEBgaKbRITE9G7d2907doVcXFxCAgIwOeff459+/a9WrAvwOSKiIiIylxmZiZ8fX2xZs0aVKtWTdyfnp6On376CUuXLkW3bt3g7u6On3/+GceOHcNff/0FANi/fz8uXryI//u//4Obmxvef/99zJkzBytXrkRubi4AICQkBC4uLliyZAkaN24Mf39/fPTRR/juu+9kvxcmV0RERFRIkGkDkJGRobfl5OQ897KjR49G79694enpqbc/NjYWGo1Gb3+jRo1Qp04dREdHAwCio6Ph6uoKOzs7sY2XlxcyMjJw4cIFsc2zfXt5eYl9yInJFREREYkUOnk2AHB0dIRarRa3BQsWFHvNLVu24NSpU8UeT0pKgqmpKaytrfX229nZISkpSWzzdGJVcLzg2IvaZGRk4J9//jH4z+lFuBQDERERlYrbt29DpVKJn83MzIptM3bsWERERECpVL7O8EoNkyuiUtB36EN8NCoFNjXy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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "tn, fp, fn, tp = matriz.ravel()\n",
        "\n",
        "precision = tp / (tp + fp)\n",
        "recall = tp / (tp + fn)\n",
        "especificidad = tn / (fp + tn)\n",
        "f1_score = 2*(precision*recall)/(precision+recall)\n",
        "\n",
        "print(f'Precision: {precision}')\n",
        "print(f'Recall: {recall}')\n",
        "print(f'Especificidad: {especificidad}')\n",
        "print(f'F1 score: {f1_score}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l5WgFqmo3b4F",
        "outputId": "e537387d-5674-4aa2-f478-540c3b1e25d9"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Precision: 1.0\n",
            "Recall: 1.0\n",
            "Especificidad: 1.0\n",
            "F1 score: 1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "#Con balanceo de clases"
      ],
      "metadata": {
        "id": "P7opVuNK48Ub"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import accuracy_score\n",
        "X_train_std1=X_train_std.copy()\n",
        "X_test_std1=X_test_std.copy()\n",
        "# Entrenamiento del modelo\n",
        "modelo1 = LogisticRegression(class_weight=\"balanced\",random_state=142)\n",
        "\n",
        "# Ajustar el modelo a los datos\n",
        "modelo1.fit(X_train_std1, y_train)\n",
        "\n",
        "# Desempeño en el entrenamiento\n",
        "y_train_pred1 = modelo1.predict(X_train_std1)\n",
        "\n",
        "print('Exactitud en el entrenamiento: %.3f'  %accuracy_score(y_train, y_train_pred1) )"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RL9berAH41uF",
        "outputId": "ce8356d6-4452-479e-debb-71c476e2d8bc"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Exactitud en el entrenamiento: 1.000\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred1 = modelo1.predict(X_test_std1)\n",
        "\n",
        "# Exactitud en el conjunto de validación\n",
        "print('Exactitud en el validacion: %.3f'  %accuracy_score(y_test, y_pred1))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Z6mpMv4PsmQT",
        "outputId": "70b47498-7c60-43d4-e982-22adb8db29f6"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Exactitud en el validacion: 1.000\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n",
        "matriz1= confusion_matrix(y_test, y_pred1)\n",
        "matriz_display1 = ConfusionMatrixDisplay(confusion_matrix=matriz1, display_labels=['No renuncia', 'renuncia'])\n",
        "matriz_display1.plot()\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 449
        },
        "id": "MCUR6vAtspIO",
        "outputId": "e7897a3b-9f3f-4978-c7f7-28bc62612b1d"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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UVkMC75pOKGdiIiISEasXBEREVEhDgtKxuSKiIiICjG5kozDgkREREQyYuWKiIiIRIr/bVL7qMyYXBEREVEhDgtKxuSKiIiIRFyKQTrOuSIiIiKSEStXREREVIjDgpIxuSIiIiJ9lTw5korDgkREREQyYuWKiIiIRJzQLh2TKyIiIirEOVeScViQiIiISEasXBEREZGIw4LSMbkiIiKiQhwWlIzDgkREREQyYuWKiIiIRBwWlI7JFRERERXisKBkTK6IiIioEJMryTjnioiIiEhGrFwRERGRiHOupGNyRURERIU4LCgZhwWJiIiIZMTKFREREYkUggCFIK30JPX8io7JFRERERXisKBkHBYkIiIikhErV0RERCTi04LSMbkiIiKiQhwWlIzDgkREREQyYuWKiIiIRBwWlI7JFRERERXisKBkTK6IiIhIxMqVdJxzRURERCQjVq6IiIioEIcFJWNyRURERHoq+7CeVBwWJCIiIpIRkysiIiIqJAjybK9o4cKFUCgUCAgIEPdlZ2dj9OjRsLW1RdWqVTFgwAAkJyfrnXfr1i307t0bFhYWqFmzJiZOnIi8vDy9NgcOHECrVq1gZmaGevXqYf369a8c54swuSIiIiJRwdOCUrdXcfLkSfz4449o3ry53v5x48Zh165d2LZtGw4ePIh79+7hww8/FI9rtVr07t0bubm5OHbsGEJDQ7F+/XoEBgaKbRITE9G7d2907doVcXFxCAgIwOeff459+/a9WrAvwOSKiIiIylxmZiZ8fX2xZs0aVKtWTdyfnp6On376CUuXLkW3bt3g7u6On3/+GceOHcNff/0FANi/fz8uXryI//u//4Obmxvef/99zJkzBytXrkRubi4AICQkBC4uLliyZAkaN24Mf39/fPTRR/juu+9kvxcmV0RERFRIkGkDkJGRobfl5OQ897KjR49G79694enpqbc/NjYWGo1Gb3+jRo1Qp04dREdHAwCio6Ph6uoKOzs7sY2XlxcyMjJw4cIFsc2zfXt5eYl9yInJFREREYkUOnk2AHB0dIRarRa3BQsWFHvNLVu24NSpU8UeT0pKgqmpKaytrfX229nZISkpSWzzdGJVcLzg2IvaZGRk4J9//jH4z+lFuBQDERERlYrbt29DpVKJn83MzIptM3bsWERERECpVL7O8EoNkyuiUtB36EN8NCoFNjXy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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Balanceo de clases manual"
      ],
      "metadata": {
        "id": "vG84ETUCxYuf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "###metricas de desempeño\n",
        "tn, fp, fn, tp = matriz1.ravel()\n",
        "\n",
        "precision = tp / (tp + fp)\n",
        "recall = tp / (tp + fn)\n",
        "especificidad = tn / (fp + tn)\n",
        "f1_score = 2*(precision*recall)/(precision+recall)\n",
        "\n",
        "print(f'Precision: {precision}')\n",
        "print(f'Recall: {recall}')\n",
        "print(f'Especificidad: {especificidad}')\n",
        "print(f'F1 score: {f1_score}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nTXpmSaIsv4Y",
        "outputId": "8e85252e-3861-4884-c22c-19ce96446123"
      },
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Precision: 1.0\n",
            "Recall: 1.0\n",
            "Especificidad: 1.0\n",
            "F1 score: 1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "os_us = SMOTETomek()\n",
        "x_train_res, y_train_res = os_us.fit_resample(X_train_std, y_train)\n",
        "\n",
        "print (\"Distribution before resampling {}\".format(Counter(y_train)))\n",
        "print (\"Distribution after resampling {}\".format(Counter(y_train_res)))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kB46ldGptBdv",
        "outputId": "a38462bc-bfab-4db3-ec4f-5f3f9ff6a051"
      },
      "execution_count": 44,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Distribution before resampling Counter({0: 14789, 1: 2851})\n",
            "Distribution after resampling Counter({0: 14789, 1: 14789})\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Selección de varianbles con el metódo Lasso"
      ],
      "metadata": {
        "id": "DtW_N69wtInJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sel_ = SelectFromModel(Lasso(alpha = 0.001, max_iter=10000), max_features=30) #entre mas aumente el parametro de serializacion, mas variables van atender a 0\n",
        "sel_.fit(X_train_std, y_train)\n",
        "print(sel_.estimator_.coef_)\n",
        "#Obtener variables seleccionadas\n",
        "X_new = sel_.get_support()#descarta los coeficientes mas cercanos a 0\n",
        "X_train_base1 = X_train_std[:,X_new]\n",
        "X_test_base1= X_test_std[:,X_new]\n",
        "X_train.iloc[:,X_new]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 649
        },
        "id": "UEfkfhBptN1Y",
        "outputId": "0967ab35-f7ae-4c95-a69c-6b0b5e7044f0"
      },
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[-0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00\n",
            " -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00\n",
            "  0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00\n",
            "  0.00000000e+00  0.00000000e+00 -3.04340773e-05 -0.00000000e+00\n",
            " -2.41766900e-05  0.00000000e+00 -9.92444916e-01  0.00000000e+00\n",
            " -0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00\n",
            " -0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00\n",
            " -0.00000000e+00 -0.00000000e+00  0.00000000e+00  0.00000000e+00\n",
            " -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00\n",
            " -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00\n",
            " -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00\n",
            "  0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00\n",
            " -0.00000000e+00  0.00000000e+00]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       trainingtimeslastyear  yearssincelastpromotion  retirementtype_No\n",
              "11163                      3                      2.0                  1\n",
              "3980                       4                      1.0                  1\n",
              "23532                      5                      7.0                  1\n",
              "19944                      2                      1.0                  1\n",
              "27056                      2                      1.0                  1\n",
              "...                      ...                      ...                ...\n",
              "2691                       3                      7.0                  1\n",
              "6807                       3                      0.0                  1\n",
              "17714                      5                      6.0                  1\n",
              "27522                      3                      0.0                  1\n",
              "9258                       5                      1.0                  1\n",
              "\n",
              "[17640 rows x 3 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-c64bb22e-826b-4217-8708-14419048664a\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>trainingtimeslastyear</th>\n",
              "      <th>yearssincelastpromotion</th>\n",
              "      <th>retirementtype_No</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>11163</th>\n",
              "      <td>3</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3980</th>\n",
              "      <td>4</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>23532</th>\n",
              "      <td>5</td>\n",
              "      <td>7.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19944</th>\n",
              "      <td>2</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27056</th>\n",
              "      <td>2</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2691</th>\n",
              "      <td>3</td>\n",
              "      <td>7.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6807</th>\n",
              "      <td>3</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17714</th>\n",
              "      <td>5</td>\n",
              "      <td>6.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27522</th>\n",
              "      <td>3</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9258</th>\n",
              "      <td>5</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>17640 rows × 3 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-c64bb22e-826b-4217-8708-14419048664a')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-c64bb22e-826b-4217-8708-14419048664a button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-c64bb22e-826b-4217-8708-14419048664a');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-778ba5cd-90fa-4178-88be-7b88a3a94f71\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-778ba5cd-90fa-4178-88be-7b88a3a94f71')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-778ba5cd-90fa-4178-88be-7b88a3a94f71 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "summary": "{\n  \"name\": \"X_train\",\n  \"rows\": 17640,\n  \"fields\": [\n    {\n      \"column\": \"trainingtimeslastyear\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 1,\n        \"min\": 0,\n        \"max\": 6,\n        \"num_unique_values\": 7,\n        \"samples\": [\n          3,\n          4,\n          6\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"yearssincelastpromotion\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 3.2289468488979325,\n        \"min\": 0.0,\n        \"max\": 15.0,\n        \"num_unique_values\": 16,\n        \"samples\": [\n          2.0,\n          1.0,\n          5.0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"retirementtype_No\",\n      \"properties\": {\n        \"dtype\": \"uint8\",\n        \"num_unique_values\": 2,\n        \"samples\": [\n          0,\n          1\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 45
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "X_train.iloc[:,X_new].columns"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dXW8pA37tZyf",
        "outputId": "0602e6b3-c1d5-402c-ac30-f778efe676d1"
      },
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Index(['trainingtimeslastyear', 'yearssincelastpromotion',\n",
              "       'retirementtype_No'],\n",
              "      dtype='object')"
            ]
          },
          "metadata": {},
          "execution_count": 46
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "##Metódo Wraper"
      ],
      "metadata": {
        "id": "RalY-oZAteFf"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Función recursiva de selección de características\n",
        "def recursive_feature_selection(X,y,model,k):\n",
        "  rfe = RFE(model, n_features_to_select=k, step=1)\n",
        "  fit = rfe.fit(X, y)\n",
        "  X_new = fit.support_\n",
        "  print(\"Num Features: %s\" % (fit.n_features_))\n",
        "  print(\"Selected Features: %s\" % (fit.support_))\n",
        "  print(\"Feature Ranking: %s\" % (fit.ranking_))\n",
        "  return X_new"
      ],
      "metadata": {
        "id": "A7k4X2Kzto8U"
      },
      "execution_count": 47,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Establecer Estimador\n",
        "model = LogisticRegression(max_iter=100)\n",
        "# Obtener columnas seleciconadas - (30 caracteristicas)\n",
        "X_new_class = recursive_feature_selection(X_train_std, y_train, model, 30)\n",
        "\n",
        "# Nuevo conjunto de datos\n",
        "X_train_base2 = X_train_std[:,X_new_class]\n",
        "X_test_base2= X_test_std[:,X_new_class]\n",
        "X_train.iloc[:,X_new_class]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "S-AittQdtuBj",
        "outputId": "648794a5-8a94-48d8-8cfc-fa01e2a20eb9"
      },
      "execution_count": 50,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n",
            "/usr/local/lib/python3.10/dist-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
            "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
            "\n",
            "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
            "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
            "Please also refer to the documentation for alternative solver options:\n",
            "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
            "  n_iter_i = _check_optimize_result(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Num Features: 30\n",
            "Selected Features: [ True  True  True False False False False  True False  True False  True\n",
            "  True False False  True False  True  True  True  True  True  True  True\n",
            "  True  True False  True False False  True False False  True  True  True\n",
            "  True False False False  True  True  True  True  True False False  True\n",
            " False  True]\n",
            "Feature Ranking: [ 1  1  1 19  3 21  9  1 13  1  4  1  1 11 18  1 16  1  1  1  1  1  1  1\n",
            "  1  1 15  1  7 10  1  8  6  1  1  1  1  2 14 17  1  1  1  1  1 12 20  1\n",
            "  5  1]\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "       environmentsatisfaction  jobsatisfaction  worklifebalance  education  \\\n",
              "11163                      4.0              4.0              3.0        3.0   \n",
              "3980                       4.0              2.0              2.0        4.0   \n",
              "23532                      4.0              4.0              3.0        5.0   \n",
              "19944                      2.0              3.0              2.0        1.0   \n",
              "27056                      3.0              1.0              3.0        2.0   \n",
              "...                        ...              ...              ...        ...   \n",
              "2691                       2.0              3.0              2.0        4.0   \n",
              "6807                       2.0              4.0              2.0        5.0   \n",
              "17714                      2.0              2.0              2.0        4.0   \n",
              "27522                      4.0              1.0              3.0        4.0   \n",
              "9258                       3.0              4.0              2.0        3.0   \n",
              "\n",
              "       monthlyincome  percentsalaryhike  stockoptionlevel  yearsatcompany  \\\n",
              "11163        88530.0               22.0               1.0               2   \n",
              "3980         25960.0               14.0               1.0              11   \n",
              "23532        83210.0               11.0               3.0               8   \n",
              "19944        24760.0               19.0               0.0               6   \n",
              "27056        29730.0               12.0               0.0              10   \n",
              "...              ...                ...               ...             ...   \n",
              "2691        182000.0               17.0               0.0               7   \n",
              "6807         26830.0               11.0               0.0               1   \n",
              "17714        24380.0               16.0               0.0               7   \n",
              "27522        55930.0               13.0               1.0               1   \n",
              "9258        142750.0               18.0               1.0               3   \n",
              "\n",
              "       retirementtype_Fired  retirementtype_No  ...  educationfield_Other  \\\n",
              "11163                     0                  1  ...                     0   \n",
              "3980                      0                  1  ...                     0   \n",
              "23532                     0                  1  ...                     0   \n",
              "19944                     0                  1  ...                     0   \n",
              "27056                     0                  1  ...                     0   \n",
              "...                     ...                ...  ...                   ...   \n",
              "2691                      0                  1  ...                     0   \n",
              "6807                      0                  1  ...                     0   \n",
              "17714                     0                  1  ...                     0   \n",
              "27522                     0                  1  ...                     1   \n",
              "9258                      0                  1  ...                     0   \n",
              "\n",
              "       educationfield_Technical Degree  gender_Female  \\\n",
              "11163                                1              0   \n",
              "3980                                 0              0   \n",
              "23532                                0              0   \n",
              "19944                                0              1   \n",
              "27056                                1              1   \n",
              "...                                ...            ...   \n",
              "2691                                 0              0   \n",
              "6807                                 1              1   \n",
              "17714                                0              0   \n",
              "27522                                0              1   \n",
              "9258                                 0              1   \n",
              "\n",
              "       jobrole_Laboratory Technician  jobrole_Manager  \\\n",
              "11163                              1                0   \n",
              "3980                               0                0   \n",
              "23532                              0                0   \n",
              "19944                              1                0   \n",
              "27056                              0                0   \n",
              "...                              ...              ...   \n",
              "2691                               0                0   \n",
              "6807                               0                0   \n",
              "17714                              0                0   \n",
              "27522                              0                0   \n",
              "9258                               0                0   \n",
              "\n",
              "       jobrole_Manufacturing Director  jobrole_Research Director  \\\n",
              "11163                               0                          0   \n",
              "3980                                0                          0   \n",
              "23532                               0                          0   \n",
              "19944                               0                          0   \n",
              "27056                               0                          0   \n",
              "...                               ...                        ...   \n",
              "2691                                0                          0   \n",
              "6807                                0                          0   \n",
              "17714                               0                          0   \n",
              "27522                               0                          0   \n",
              "9258                                0                          0   \n",
              "\n",
              "       jobrole_Research Scientist  maritalstatus_Divorced  \\\n",
              "11163                           0                       0   \n",
              "3980                            0                       0   \n",
              "23532                           0                       0   \n",
              "19944                           0                       0   \n",
              "27056                           1                       0   \n",
              "...                           ...                     ...   \n",
              "2691                            1                       0   \n",
              "6807                            0                       1   \n",
              "17714                           0                       1   \n",
              "27522                           0                       1   \n",
              "9258                            1                       0   \n",
              "\n",
              "       maritalstatus_Single  \n",
              "11163                     0  \n",
              "3980                      1  \n",
              "23532                     0  \n",
              "19944                     0  \n",
              "27056                     1  \n",
              "...                     ...  \n",
              "2691                      1  \n",
              "6807                      0  \n",
              "17714                     0  \n",
              "27522                     0  \n",
              "9258                      1  \n",
              "\n",
              "[17640 rows x 30 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-d80da7d0-1b8d-4da4-994e-a85b0e58f0b9\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>environmentsatisfaction</th>\n",
              "      <th>jobsatisfaction</th>\n",
              "      <th>worklifebalance</th>\n",
              "      <th>education</th>\n",
              "      <th>monthlyincome</th>\n",
              "      <th>percentsalaryhike</th>\n",
              "      <th>stockoptionlevel</th>\n",
              "      <th>yearsatcompany</th>\n",
              "      <th>retirementtype_Fired</th>\n",
              "      <th>retirementtype_No</th>\n",
              "      <th>...</th>\n",
              "      <th>educationfield_Other</th>\n",
              "      <th>educationfield_Technical Degree</th>\n",
              "      <th>gender_Female</th>\n",
              "      <th>jobrole_Laboratory Technician</th>\n",
              "      <th>jobrole_Manager</th>\n",
              "      <th>jobrole_Manufacturing Director</th>\n",
              "      <th>jobrole_Research Director</th>\n",
              "      <th>jobrole_Research Scientist</th>\n",
              "      <th>maritalstatus_Divorced</th>\n",
              "      <th>maritalstatus_Single</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>11163</th>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>88530.0</td>\n",
              "      <td>22.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>2</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3980</th>\n",
              "      <td>4.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>25960.0</td>\n",
              "      <td>14.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>11</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>23532</th>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>83210.0</td>\n",
              "      <td>11.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>8</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>19944</th>\n",
              "      <td>2.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>24760.0</td>\n",
              "      <td>19.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>6</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27056</th>\n",
              "      <td>3.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>29730.0</td>\n",
              "      <td>12.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2691</th>\n",
              "      <td>2.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>182000.0</td>\n",
              "      <td>17.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6807</th>\n",
              "      <td>2.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>5.0</td>\n",
              "      <td>26830.0</td>\n",
              "      <td>11.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>17714</th>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>24380.0</td>\n",
              "      <td>16.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>27522</th>\n",
              "      <td>4.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>55930.0</td>\n",
              "      <td>13.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>9258</th>\n",
              "      <td>3.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>142750.0</td>\n",
              "      <td>18.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>...</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>17640 rows × 30 columns</p>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-d80da7d0-1b8d-4da4-994e-a85b0e58f0b9')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-d80da7d0-1b8d-4da4-994e-a85b0e58f0b9 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-d80da7d0-1b8d-4da4-994e-a85b0e58f0b9');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-0ac315e7-c8dd-42ee-9ca9-7c04b7d5db70\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-0ac315e7-c8dd-42ee-9ca9-7c04b7d5db70')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-0ac315e7-c8dd-42ee-9ca9-7c04b7d5db70 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe"
            }
          },
          "metadata": {},
          "execution_count": 50
        }
      ]
    }
  ]
}