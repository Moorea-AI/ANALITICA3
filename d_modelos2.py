{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOgeoSeoZczRtFIoCCl1dzI",
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
      "execution_count": 45,
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
      "execution_count": 46,
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
      "execution_count": 47,
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
        "outputId": "801bcaa0-53e0-47de-db0a-36b39b87eb2a"
      },
      "execution_count": 48,
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
      "execution_count": 49,
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
      "execution_count": 50,
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
      "execution_count": 51,
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
        "outputId": "13a21399-3973-4bb2-d748-6e3da5f49d40"
      },
      "execution_count": 52,
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
      "execution_count": 53,
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
        "outputId": "d3ce19d0-e89f-4c5d-ffcb-d7df033c53c0"
      },
      "execution_count": 54,
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
        "outputId": "0c70e0b5-90bc-4f53-e779-21c03b871178"
      },
      "execution_count": 55,
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
      "execution_count": 57,
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
      "execution_count": 58,
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
      "execution_count": 60,
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
        "outputId": "5fcb16a8-9921-4001-d097-9ba1a5643ec3"
      },
      "execution_count": 61,
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
        "outputId": "d0ddbce8-0f67-4c8c-cb95-7a17cdb4f584"
      },
      "execution_count": 62,
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
        "outputId": "555ca433-76bd-4fb6-d446-336dfae9528f"
      },
      "execution_count": 63,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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\n"
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
        "outputId": "7dfd4f3f-ac83-4fc2-85c2-302e1086bdad"
      },
      "execution_count": 64,
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
        "outputId": "72fb2615-7bdc-4458-a3f9-8a4fc75b7b33"
      },
      "execution_count": 65,
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
    }
  ]
}