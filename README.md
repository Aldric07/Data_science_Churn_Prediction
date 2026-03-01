readme_fr = """
# Prédiction du Churn Client – Télécommunications

## Présentation du Projet

Ce projet a pour objectif de prédire le **churn client** (résiliation) dans une entreprise de télécommunications à l’aide de techniques de Machine Learning.

Le churn correspond aux clients quittant les services de l’entreprise. Sa prédiction permet de mettre en place des stratégies de rétention et de limiter les pertes financières.

Le projet suit un workflow complet de **Data Science** :

Exploration des données  
Prétraitement des données  
Apprentissage non supervisé (segmentation clients)  
Apprentissage supervisé (modèles de prédiction)  
Comparaison des modèles  
Déploiement via API FastAPI  

---

## Objectif Métier

Identifier les clients susceptibles de quitter l’entreprise afin de :

- Améliorer la fidélisation client
- Réduire les pertes de revenus
- Cibler efficacement les campagnes marketing
- Comprendre les comportements clients

---

##  Structure du Projet

.
├── eda.ipynb
│   
│
├── models/
│   ├── logreg.pkl
│   ├── random_forest.pkl
│   └── naive_bayes.pkl
│
├── api/
│   └── main.py
│
├── requirements.txt
└── README.md

---

##  Description du Dataset

Le jeu de données contient des informations relatives aux clients :

- Données démographiques
- Informations contractuelles
- Utilisation des services Internet
- Facturation
- Type d’abonnement

###  Variable cible

`Customer Status`

Transformée en classification binaire :

| Valeur | Signification |
|------|------|
| 0 | Client fidèle |
| 1 | Client en churn |

---

## ⚙️ Workflow Data Science

### 1️ Analyse Exploratoire (EDA)

Analyses réalisées :

- Structure du dataset
- Analyse des valeurs manquantes
- Distribution des variables
- Répartition du churn
- Relation entre services et churn

Insight principal :
> Les contrats mensuels présentent un taux de churn plus élevé.

---

### 2️⃣ Prétraitement des Données

Étapes appliquées :

- Gestion des valeurs manquantes
- Encodage des variables catégorielles (OneHotEncoder)
- Normalisation / standardisation (StandardScaler)
- Pipelines Scikit-learn pour éviter le data leakage

La standardisation est appliquée uniquement lorsque nécessaire :
- ✅ Régression Logistique
- ✅ Naive Bayes  
- ❌ Random Forest

---

### 3️⃣ Apprentissage Non Supervisé – Segmentation Clients

Un algorithme **K-Means** a été utilisé pour identifier des groupes naturels de clients.

Étapes :
- Prétraitement des features
- Choix optimal du nombre de clusters (Silhouette Score)
- Visualisation PCA des clusters
- Interprétation métier des segments
- Analyse du churn par cluster

Insight :
> Certains segments présentent un risque de churn significativement plus élevé.

---

### 4️⃣ Modèles Supervisés

Trois modèles de classification ont été implémentés.

#### 🔹 Régression Logistique
- Modèle linéaire interprétable
- Prédiction probabiliste

**Métrique principale : Recall**
→ éviter de manquer des clients à risque.

---

#### 🌲 Random Forest
- Ensemble d’arbres de décision
- Capture les relations non linéaires

**Métrique principale : F1-score**
→ équilibre entre précision et rappel.

---

#### 🧮 Naive Bayes (GaussianNB)
- Classificateur probabiliste rapide

**Métrique principale : Precision**
→ limitation des faux positifs.

---

### 5️⃣ Comparaison des Modèles

Les modèles ont été évalués avec :

- Precision
- Recall
- F1-score
- ROC-AUC

Le modèle Random Forest présente les performances globales les plus équilibrées.

---

## Déploiement – API FastAPI

Une API REST a été développée pour exposer les modèles entraînés.

### Fonctionnalités

Chargement simultané des modèles  
Sélection dynamique du modèle  
Validation des entrées  
Retour de probabilité de churn  

---

## ▶️ Lancer l’API

### Installer les dépendances
pip install -r requirements.txt

### Démarrer le serveur
uvicorn api.main:app --reload

---

##  Endpoints API

### Vérification API
GET /health

### Liste des modèles
GET /models

### Prédiction churn
POST /predict?model_name=random_forest

Modèles disponibles :
- logreg
- random_forest
- naive_bayes

---

##  Exemple de réponse

{
  "model": "random_forest",
  "churn_prediction": 1,
  "churn_probability": 0.78
}

---

## 🧠 Impact Métier

Ce système permet :

- La détection précoce des clients à risque
- La mise en place d’actions de rétention ciblées
- La réduction des pertes financières
- L’amélioration de la satisfaction client

---

## ⚠️ Limites

- Données historiques uniquement
- Dépendance à la qualité des variables
- Facteurs externes non pris en compte

---

##  Améliorations Futures

- Prédiction en temps réel
- Monitoring des modèles
- Réentraînement automatique
- Déploiement Cloud / Docker
- Explicabilité des modèles (SHAP)

---

##  Technologies Utilisées

- Python
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn
- FastAPI
- Cloudpickle
- Jupyter Notebook

---

## 📜 Auteur
Réalisé par : Yoann DOSSOU-YOVO et Cheick COULIBALY
Projet Data Science – Prédiction du Churn Client  
EFREI Paris – Master Data Engineering & Intelligence Artificielle
"""

file_path = Path("/mnt/data/README_FR.md")
file_path.write_text(readme_fr)

str(file_path)