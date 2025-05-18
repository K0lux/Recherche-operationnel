# Simplex RO – Application d’Optimisation et d’Algorithmes de Graphes

Ce projet est une application web Flask permettant :
- La résolution de problèmes de programmation linéaire (méthode du Simplex)
- L’analyse de solutions (sensibilité, dualité…)
- L’historique des calculs
- La visualisation et l’exécution d’algorithmes de graphes (Dijkstra, Prim, Kruskal, Ford-Fulkerson) à partir d’une matrice d’adjacence

---

## Installation

### 1. Cloner le dépôt
```bash
git clone <url-du-repo>
cd RPO/simplex_app
```

### 2. Installer les dépendances Python
Assurez-vous d’avoir Python 3.8+ installé.

```bash
python -m venv venv
venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
```

> Si le fichier `requirements.txt` n’existe pas, créez-le avec :
> ```
> Flask
> ````
> Et ajoutez d’autres dépendances si besoin (vis-network est chargé côté JS).

### 3. Lancer l’application

```bash
python app.py
```

L’application sera accessible sur http://127.0.0.1:5000

---

## Utilisation

### 1. Calculateur Simplex
- Accédez à l’onglet « Calculateur Simplex »
- Saisissez la fonction objectif, les contraintes et lancez le calcul
- Les résultats détaillés s’affichent, avec l’analyse de sensibilité

### 2. Historique
- Toutes les résolutions sont enregistrées
- Vous pouvez consulter les anciens calculs dans l’onglet « Historique »

### 3. Algorithmes de Graphes
- Allez dans l’onglet « Algorithmes Graphes »
- Définissez le nombre de nœuds, leurs noms, et la matrice d’adjacence (poids des arêtes)
- Cochez « Graphe dirigé » si besoin
- Remplissez les paramètres pour Dijkstra, Prim, Ford-Fulkerson
- Cliquez sur « Exécuter et Visualiser »
- Le graphe s’affiche dynamiquement (vis.js) et les résultats des algorithmes sont donnés en bas de page

---

## Structure du projet

```
simplex_app/
├── app.py                  # Application Flask principale
├── requirements.txt        # Dépendances Python
├── app/
│   ├── __init__.py
│   ├── simplex_solver.py
│   └── problem_parser.py
├── static/
│   ├── style.css           # Styles CSS
│   └── graphs.js           # JS pour la visualisation des graphes
├── templates/
│   ├── index.html          # Page d’accueil/calculateur
│   ├── graphs.html         # Page des algorithmes de graphes
│   ├── history.html        # Historique des calculs
│   └── navbar.html         # Barre de navigation commune
└── db/
    ├── __init__.py
    ├── schema.sql          # base de donnée sqllite
    └── db_manager.py       # Gestion de la base SQLite (historique)
```

---

## Conseils
- Pour toute erreur, consultez la console Flask et la console du navigateur (F12)
- Les algorithmes de graphes attendent des entrées cohérentes (noms de nœuds, matrice symétrique pour les graphes non dirigés…)
- Le style de la navbar est personnalisable dans `static/style.css`

---

## Auteurs
- Inspiré par les cours de Recherche Opérationnelle
- Développé par KASSA Malipita Luc

---

## Licence
Ce projet est fourni à des fins pédagogiques.
