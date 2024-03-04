{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPXDxuJeLlE2RbF31iG+96e",
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
      "execution_count": 23,
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
        "from sklearn.linear_model import LogisticRegression"
      ],
      "metadata": {
        "id": "k8_bX0o1gtD_"
      },
      "execution_count": 24,
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
      "execution_count": 25,
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
        "outputId": "64d81403-4d58-454c-9364-3db025a037b1"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
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
      "execution_count": 27,
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
      "execution_count": 28,
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
      "execution_count": 29,
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
        "outputId": "4164bea9-4754-43ca-9251-8f38699d17d4"
      },
      "execution_count": 30,
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
      "execution_count": 31,
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
        "outputId": "2429ac48-4f44-4f4d-e8e2-04e1a63d9077"
      },
      "execution_count": 32,
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
        "outputId": "0e031c88-9ebd-43c8-d6a6-a634ebba55e0"
      },
      "execution_count": 33,
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
        "df_d = df_dummies.drop(['datesurvey', 'surveydate', 'infodate', 'employeeid', 'retirementdate'],  axis=1)"
      ],
      "metadata": {
        "id": "HJSzqqGc9GlY"
      },
      "execution_count": 34,
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
      "execution_count": 35,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "y=df_d[\"attrition\"]\n",
        "x=df_d.drop([\"attrition\"],axis=1)\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)\n",
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
      "execution_count": 36,
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
        "modelo = LogisticRegression(max_iter=20000, solver='sag')\n",
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
        "outputId": "b84765a3-9061-4d5f-f5d1-d91917bd0641"
      },
      "execution_count": 39,
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
        "outputId": "bf8eecf6-20df-4d1f-dee7-fd6c763b303d"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Exactitud en la validacion: 1.000\n"
          ]
        }
      ]
    }
  ]
}