{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOpTjRmNn29cHnPkJ55xoSs",
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
        "<a href=\"https://colab.research.google.com/github/Moorea-AI/ANALITICA3/blob/main/b_preprocesamiento.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def ejecutar_sql (nombre_archivo, cur):\n",
        "  sql_file=open(nombre_archivo)\n",
        "  sql_as_string=sql_file.read()\n",
        "  sql_file.close\n",
        "  cur.executescript(sql_as_string)"
      ],
      "metadata": {
        "id": "sNztg3cvjSDP"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "sp9XXeKSrFPU"
      },
      "outputs": [],
      "source": [
        "#### Cargar paquetes siempre al inicio\n",
        "import pandas as pd ### para manejo de datos\n",
        "import sqlite3 as sql #### para bases de datos sql\n",
        "import sys ## saber ruta de la que carga paquetes\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "qS-Z8OP0sEcE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "ejecutar_sql"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 104
        },
        "id": "_biBJDXtioT3",
        "outputId": "8fb4235a-37bb-4ea6-d1f5-5a4a2a9d10d2"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function __main__.ejecutar_sql(nombre_archivo, cur)>"
            ],
            "text/html": [
              "<div style=\"max-width:800px; border: 1px solid var(--colab-border-color);\"><style>\n",
              "      pre.function-repr-contents {\n",
              "        overflow-x: auto;\n",
              "        padding: 8px 12px;\n",
              "        max-height: 500px;\n",
              "      }\n",
              "\n",
              "      pre.function-repr-contents.function-repr-contents-collapsed {\n",
              "        cursor: pointer;\n",
              "        max-height: 100px;\n",
              "      }\n",
              "    </style>\n",
              "    <pre style=\"white-space: initial; background:\n",
              "         var(--colab-secondary-surface-color); padding: 8px 12px;\n",
              "         border-bottom: 1px solid var(--colab-border-color);\"><b>ejecutar_sql</b><br/>def ejecutar_sql(nombre_archivo, cur)</pre><pre class=\"function-repr-contents function-repr-contents-collapsed\" style=\"\"><a class=\"filepath\" style=\"display:none\" href=\"#\">/content/&lt;ipython-input-4-91ed477acf2a&gt;</a>&lt;no docstring&gt;</pre></div>"
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "employee = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/RH_Grupo1/databases/employee_survey_data.csv'\n",
        "general = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/RH_Grupo1/databases/general_data.csv'\n",
        "manager = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/RH_Grupo1/databases/manager_survey.csv'\n",
        "retirement = 'https://raw.githubusercontent.com/Moorea-AI/ANALITICA3/main/RH_Grupo1/databases/retirement_info.csv'"
      ],
      "metadata": {
        "id": "OQrmPsjOlph2"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df_employee=pd.read_csv(employee)\n",
        "df_general=pd.read_csv(general)\n",
        "df_manager=pd.read_csv(manager)\n",
        "df_retirement=pd.read_csv(retirement)"
      ],
      "metadata": {
        "id": "yKMQExbaloc4"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "conn= sql.connect(\"data\\\\db_empleados\") ### crea una base de datos con el nombre dentro de comillas, si existe crea una conexión.\n",
        "cur=conn.cursor() ### ejecutar funciones  en BD\n"
      ],
      "metadata": {
        "id": "wKI8QqO3m5hY"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "### Llevar tablas a base de datos\n",
        "df_employee.to_sql(\"employee\",conn,if_exists=\"replace\")\n",
        "df_general.to_sql(\"general\",conn,if_exists=\"replace\")\n",
        "df_manager.to_sql(\"manager\",conn,if_exists=\"replace\")\n",
        "df_retirement.to_sql(\"retirement\",conn,if_exists=\"replace\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "dKqHj3GFquXy",
        "outputId": "4712be07-b15e-4437-94e3-27fb6bd9c0f3"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "711"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    }
  ]
}