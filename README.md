# 🧠 Stroke Bias Analysis
### Application de Détection de Biais dans la Prédiction du Risque d'AVC

> **Parcours A — Détection de Biais** | Dataset : [Stroke Prediction (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---

## 📋 Table des Matières

1. [Présentation](#-présentation)
2. [Fonctionnalités](#-fonctionnalités)
3. [Architecture & Méthodes](#-architecture--méthodes)
4. [Installation & Lancement](#-installation--lancement)
5. [Guide d'utilisation page par page](#-guide-dutilisation-page-par-page)
6. [Structure du projet](#-structure-du-projet)
7. [Dépendances](#-dépendances)
8. [Déploiement en ligne](#-déploiement-en-ligne-streamlit-cloud)
9. [FAQ](#-faq)

---

## 🎯 Présentation

**Stroke Bias Analysis** est une application interactive construite avec [Streamlit](https://streamlit.io) qui permet d'explorer, analyser et quantifier les biais algorithmiques dans la prédiction du risque d'AVC (accident vasculaire cérébral).

### Pourquoi ce sujet ?

Les modèles de machine learning entraînés sur des données médicales peuvent reproduire — voire amplifier — des inégalités existantes. Si un modèle prédit différemment selon le **genre** (Homme / Femme) ou la **zone géographique** (Rural / Urban), des groupes entiers de patients pourraient être sous-diagnostiqués, avec des conséquences directes sur leur prise en charge.

### Dataset utilisé

| Propriété | Valeur |
|-----------|--------|
| Source | Kaggle — fedesoriano |
| Taille | 5 109 patients × 12 colonnes |
| Variable cible | `stroke` (0 = pas d'AVC, 1 = AVC) |
| Taux d'AVC | 4,87 % (dataset fortement déséquilibré) |
| Valeurs manquantes | 201 valeurs manquantes dans `bmi` (imputées par la médiane) |
| Attributs sensibles | `gender` (Male/Female) et `Residence_type` (Rural/Urban) |

---

## ✨ Fonctionnalités

| Page | Fonctionnalité | Description |
|------|---------------|-------------|
| 🏠 **Accueil** | Vue d'ensemble | KPIs, contexte, aperçu des données, description des colonnes |
| 📊 **Exploration** | Analyse descriptive | 4 visualisations interactives, heatmap de corrélations, dataframe complet |
| ⚠️ **Détection de Biais** | Fairness sur les données brutes | Métriques de parité démographique et d'impact disproportionné |
| 🤖 **Modélisation** | Évaluation de l'équité du modèle | Entraînement, performances globales et par groupe, matrices de confusion |
| 🎯 **Prédiction** | Prédiction individuelle | Formulaire patient, score de risque 0-100, risque relatif, jauge interactive |

---

## 🔬 Architecture & Méthodes

### Prétraitement des données

```
1. Suppression de la ligne "Other" dans `gender` (1 seule occurrence)
2. Imputation des valeurs manquantes de `bmi` par la médiane (201 valeurs)
3. Encodage Label des variables catégorielles pour la modélisation
```

### Métriques de Fairness (utils/fairness.py)

Trois métriques sont calculées, conformes aux standards IEEE/ACM de l'équité algorithmique :

#### 1. Différence de Parité Démographique (Demographic Parity Difference)
```
DPD = P(Ŷ=1 | groupe A) − P(Ŷ=1 | groupe B)
```
- **Interprétation** : mesure l'écart de taux de prédiction positive entre deux groupes
- **Seuil acceptable** : |DPD| < 0,05
- **Valeur idéale** : 0 (même taux pour tous les groupes)

#### 2. Ratio d'Impact Disproportionné (Disparate Impact Ratio)
```
DI = P(Ŷ=1 | groupe non-privilégié) / P(Ŷ=1 | groupe privilégié)
```
- **Interprétation** : ratio des taux entre groupes
- **Règle des 4/5** : DI ≥ 0,8 requis pour absence de biais légal
- **Valeur idéale** : 1 (ratio égal)

#### 3. Différence d'Égalité des Chances (Equalized Odds Difference)
```
EOD = |TPR_groupe_A − TPR_groupe_B|
```
- **Interprétation** : mesure si le modèle détecte les vrais positifs également bien dans chaque groupe
- **Valeur idéale** : 0

### Modèles disponibles

| Modèle | Usage | Paramètres clés |
|--------|-------|----------------|
| **Logistic Regression** | Page Modélisation | `class_weight='balanced'`, `max_iter=1000` |
| **Random Forest** | Page Modélisation | `n_estimators=100`, `class_weight='balanced'` |
| **Gradient Boosting** | Page Prédiction | `n_estimators=200`, `lr=0.05`, `max_depth=4` |

> ⚠️ **Pourquoi `class_weight='balanced'` ?**
> Le dataset est fortement déséquilibré (4,87 % d'AVC). Sans correction, le modèle prédit "pas d'AVC" pour tous les patients et obtient 95% d'accuracy mais 0% de recall. Le paramètre `balanced` corrige ce biais de classe.

### Score de Risque Relatif (Page Prédiction)

La page Prédiction n'affiche pas la probabilité brute (qui serait toujours ≤ 10% vu le taux de base), mais un **score relatif à la population** :

```
score = min(100, (P(AVC | patient) / P95_population) × 100)
risque_relatif = P(AVC | patient) / taux_base
```

- **P95** : seuil au 95e percentile des probabilités sur toute la population
- Un score de 100/100 signifie que le patient est dans le top 5% des profils les plus à risque
- Le risque relatif ("3× la moyenne") est plus interprétable médicalement qu'une probabilité brute de 9%

---

## 🚀 Installation & Lancement

### Prérequis

- Python 3.9 ou supérieur
- pip

### Étape 1 — Cloner ou télécharger le projet

```bash
# Option A : depuis GitHub
git clone https://github.com/votre-compte/stroke-bias-analysis.git
cd stroke-bias-analysis

# Option B : décompresser l'archive téléchargée
unzip stroke_app_v3.zip
cd stroke_app_v3
```

### Étape 2 — Créer un environnement virtuel (recommandé)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 — Vérifier la présence du dataset

Assurez-vous que le fichier `healthcare-dataset-stroke-data.csv` est bien dans le **même dossier** que `app.py` :

```
stroke_app_v3/
├── app.py                              ← fichier principal
├── healthcare-dataset-stroke-data.csv  ← REQUIS ici
├── requirements.txt
├── README.md
└── utils/
    ├── __init__.py
    └── fairness.py
```

> Si le CSV est absent, téléchargez-le sur [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) et placez-le dans ce dossier.

### Étape 5 — Lancer l'application

```bash
# IMPORTANT : lancer depuis le dossier stroke_app_v3/
cd stroke_app_v3
streamlit run app.py
```

L'application s'ouvre automatiquement sur **http://localhost:8501**

---

## 📖 Guide d'utilisation page par page

### 🏠 Page 1 — Accueil

La page d'accueil présente le contexte général du projet.

**Ce que vous y trouvez :**
- 4 métriques clés : nombre de patients, colonnes, taux de valeurs manquantes, taux d'AVC
- Un texte de contextualisation sur les biais dans les modèles médicaux
- Un aperçu interactif des premières lignes du dataset (scrollable)
- Un tableau descriptif de chaque variable avec son type

**Astuce :** Cliquez sur les en-têtes de colonnes du tableau pour trier les données.

---

### 📊 Page 2 — Exploration des Données

Page d'analyse exploratoire avec 4 visualisations interactives.

**Visualisations disponibles :**

| # | Titre | Description |
|---|-------|-------------|
| 1 | Distribution de la Variable Cible | Barres + camembert de la répartition AVC / Pas d'AVC |
| 2 | Comparaison par Attribut Sensible | Effectifs AVC par genre ET par zone géographique |
| 3 | Heatmap des Corrélations | Matrice de corrélation entre toutes les variables numériques |
| 4 | Données brutes | Tableau complet filtrable et triable |

**À noter :** Les graphiques Plotly sont interactifs — vous pouvez zoomer, cliquer sur la légende pour masquer/afficher des séries, et télécharger chaque graphique en PNG via l'icône appareil photo.

---

### ⚠️ Page 3 — Détection de Biais

**C'est la page centrale du projet.** Elle analyse les biais dans les données brutes, avant toute modélisation.

**Comment utiliser :**

1. **Choisissez l'attribut sensible** dans le menu déroulant :
   - `Genre (gender)` → analyse Homme vs Femme
   - `Zone géographique (Residence_type)` → analyse Rural vs Urban

2. **Lisez les 4 sections** qui s'affichent :

   - **Section 1 — Explication** : contexte du biais choisi et pourquoi il est problématique
   - **Section 2 — Métriques** : deux métriques de fairness calculées sur les données brutes
     - ✅ Vert : pas de biais détecté
     - ⚠️ Jaune : biais modéré à surveiller
     - 🚨 Rouge : biais significatif
   - **Section 3 — Visualisation** : graphique des taux d'AVC par groupe + répartition de la population
   - **Section 4 — Interprétation** : analyse du groupe défavorisé et recommandations concrètes

**Résultats attendus :**

| Attribut | DPD | Ratio DI | Interprétation |
|----------|-----|----------|----------------|
| Genre | 0,0040 | 0,9223 | Règle des 4/5 respectée ✅ |
| Zone géo | 0,0066 | 0,8723 | Règle des 4/5 respectée ✅ |

---

### 🤖 Page 4 — Modélisation

Entraîne un modèle de machine learning et évalue son équité.

**Comment utiliser :**

1. **Sélectionnez l'algorithme** : Logistic Regression ou Random Forest
2. **Sélectionnez l'attribut sensible** à évaluer : `gender` ou `Residence_type`
3. Attendez le chargement (le modèle s'entraîne en temps réel, ~2-5 secondes)

**Ce que vous obtenez :**

- **Performances globales** : Accuracy, Precision, Recall, F1-Score
- **Métriques de fairness sur les prédictions** : DPD, Ratio DI, Equalized Odds
- **Tableau de performances par groupe** : comparaison des métriques Homme vs Femme (ou Rural vs Urban)
- **Graphique de comparaison** : barres côte à côte par groupe
- **Matrices de confusion** : une par groupe pour visualiser les erreurs
- **Importance des variables** (Random Forest uniquement)

**Performances de référence (Logistic Regression, dataset test 20%) :**

| Métrique | Valeur |
|----------|--------|
| Accuracy | 0,736 |
| Recall | 0,800 |
| AUC | ~0,82 |

---

### 🎯 Page 5 — Prédiction

Outil de prédiction individuelle du risque d'AVC pour un patient fictif.

**Comment utiliser :**

1. **Renseignez les paramètres du patient** via les 10 contrôles :

   | Contrôle | Type | Plage |
   |----------|------|-------|
   | Âge | Slider | 1 – 82 ans |
   | Genre | Sélecteur | Male / Female |
   | Hypertension | Sélecteur | Non / Oui |
   | Maladie cardiaque | Sélecteur | Non / Oui |
   | Déjà marié(e) | Sélecteur | Yes / No |
   | Type de travail | Sélecteur | Private / Self-employed / Govt_job / children / Never_worked |
   | Zone géographique | Sélecteur | Urban / Rural |
   | Glucose moyen | Slider | 55 – 272 mg/dL |
   | IMC (BMI) | Slider | 10 – 98 |
   | Statut tabagique | Sélecteur | never smoked / formerly smoked / smokes / Unknown |

2. **Cliquez sur "🔍 Prédire le Risque d'AVC"**

3. **Interprétez les résultats :**

   - **Score de risque (0-100)** : position du patient dans la distribution de la population
     - 0-25 : risque faible ✅
     - 25-55 : risque modéré ⚠️
     - 55+ : risque élevé 🚨
   - **Risque relatif** : combien de fois plus à risque que la moyenne de la population
   - **Jauge colorée** : visualisation immédiate du niveau de risque
   - **Tableau comparatif** : valeurs du patient vs moyennes de la population

> ⚠️ **Avertissement** : Cet outil est à des fins éducatives uniquement et ne constitue pas un diagnostic médical.

**Exemple de profils types :**

| Profil | Score | Risque relatif |
|--------|-------|----------------|
| Homme, 75 ans, hypertension, maladie cardiaque, glucose 220, fumeur | ~100/100 | ~6,5× |
| Homme, 55 ans, hypertension, glucose 150 | ~53/100 | ~2,3× |
| Femme, 25 ans, aucun facteur de risque | ~1/100 | ~0× |

---

## 📁 Structure du Projet

```
stroke_app_v3/
│
├── app.py                               # Application principale Streamlit (5 pages)
│
├── utils/
│   ├── __init__.py                      # Rend utils importable comme module Python
│   └── fairness.py                      # Fonctions de métriques d'équité algorithmique
│                                        #   · demographic_parity_difference()
│                                        #   · disparate_impact_ratio()
│                                        #   · equalized_odds_difference()
│
├── healthcare-dataset-stroke-data.csv   # Dataset (à télécharger sur Kaggle)
├── requirements.txt                     # Dépendances Python
└── README.md                            # Ce fichier
```

---

## 📦 Dépendances

```
streamlit>=1.32.0       # Framework web interactif
pandas>=2.0.0           # Manipulation de données
numpy>=1.26.0           # Calculs numériques
plotly>=5.18.0          # Graphiques interactifs
scikit-learn>=1.4.0     # Modèles ML et métriques
```

Installation :
```bash
pip install -r requirements.txt
```

---

## ☁️ Déploiement en ligne (Streamlit Cloud)

Pour partager l'application sans installation locale :

### Étape 1 — Préparer le dépôt GitHub

```bash
git init
git add .
git commit -m "Initial commit — Stroke Bias Analysis"
git remote add origin https://github.com/votre-compte/stroke-bias-analysis.git
git push -u origin main
```

> ⚠️ Assurez-vous que le fichier `healthcare-dataset-stroke-data.csv` est bien inclus dans le dépôt.

### Étape 2 — Déployer sur Streamlit Cloud

1. Rendez-vous sur [share.streamlit.io](https://share.streamlit.io)
2. Connectez votre compte GitHub
3. Cliquez sur **"New app"**
4. Sélectionnez votre dépôt et la branche `main`
5. Dans **"Main file path"**, entrez : `app.py`
6. Cliquez sur **"Deploy"**

L'application sera accessible via une URL publique du type :
```
https://votre-compte-stroke-bias-analysis-app-xxxx.streamlit.app
```

---

## ❓ FAQ

**Q : L'application ne démarre pas avec l'erreur `ModuleNotFoundError: No module named 'utils.fairness'`**

> Vous devez lancer Streamlit **depuis l'intérieur** du dossier `stroke_app_v3/`, pas depuis le dossier parent.
> ```bash
> cd stroke_app_v3
> streamlit run app.py   # ✅ correct
> ```
> ```bash
> streamlit run stroke_app_v3/app.py  # ❌ incorrect
> ```

---

**Q : Pourquoi les probabilités d'AVC sont-elles si basses (< 10%) ?**

> C'est statistiquement correct. Le taux d'AVC dans la population est de 4,87%. Même pour un patient à très haut risque, la probabilité absolue reste modérée. C'est pourquoi la page Prédiction affiche un **score relatif** (0-100) plutôt qu'une probabilité brute — il est bien plus interprétable.

---

**Q : Pourquoi utiliser `class_weight='balanced'` dans les modèles ?**

> Sans cette correction, les modèles prédisent "pas d'AVC" pour 100% des patients (ce qui donne une accuracy trompeuse de 95% mais un recall de 0%). Le paramètre `balanced` donne plus de poids aux cas d'AVC rares lors de l'entraînement, permettant au modèle de réellement détecter les cas positifs.

---

**Q : Quelle est la différence entre la page Détection de Biais et la page Modélisation ?**

> - **Détection de Biais** : analyse les biais dans les **données brutes** (avant modélisation). Elle montre si certains groupes sont sur/sous-représentés dans les données elles-mêmes.
> - **Modélisation** : analyse si le **modèle entraîné** reproduit ou amplifie ces biais sur ses prédictions.

---

**Q : Le modèle de la page Prédiction est-il le même que celui de la page Modélisation ?**

> Non. La page Modélisation utilise Logistic Regression ou Random Forest (au choix). La page Prédiction utilise un **Gradient Boosting** (GBM) qui offre un meilleur AUC (0,81) et des probabilités mieux étalées, plus adaptées à l'outil de prédiction individuelle.

---

## 👤 Auteur

Projet réalisé dans le cadre du **Parcours A — Détection de Biais**.
Dataset : [Stroke Prediction Dataset — fedesoriano (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
