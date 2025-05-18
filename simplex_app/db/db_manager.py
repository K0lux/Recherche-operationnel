import sqlite3
import json # Import json here as well if you need helper functions in db.py

# Dans simplex_app/db/db_manager.py
#with current_app.open_resource('schema.sql') as f:
    #db.executescript(f.read().decode('utf8'))
# devrait toujours fonctionner si 'schema.sql' est dans le dossier 'db' à côté de db_manager.py
# OU vous pouvez utiliser des chemins absolus ou relatifs basés sur __file__

import click
from flask import current_app, g

def get_db():
    """Connects to the application's configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row # Return rows that behave like dicts

    return g.db

def close_db(e=None):
    """Closes the database again at the end of the request."""
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    """Clears existing data and creates new tables."""
    db = get_db()

    with current_app.open_resource('db/schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
def init_db_command():
    """Commande CLI pour initialiser la base de données."""
    init_db()
    click.echo('Initialized the database.')

def init_app(app):
    """Enregistre les fonctions de gestion de la base de données auprès de l’app Flask.This is called by
 the application factory.
    """
    app.teardown_appcontext(close_db) # Close DB after each request
    app.cli.add_command(init_db_command) # Add the init-db command

def add_history_entry(entry_type, problem_data, results_data):
    """
    Adds a completed calculation to the history database.
    entry_type: 'simplex' or 'graph'
    problem_data: dict containing the input data for the problem
    results_data: dict containing the results (status, values, etc.)
    """
    db = get_db()
    
    # Initialiser tous les champs potentiels à None
    obj_type, obj_coeffs_json, constraints_json = None, None, None
    graph_input_type, graph_data_json, graph_is_directed_int = None, None, None
    graph_params_json, graph_results_json = None, None
    status, objective_val, solution_vars_json = None, None, None
    warning_msg = results_data.get('warning_message') # Warning est commun

    status = results_data.get('status', 'Inconnu')

    if entry_type == 'simplex':
        obj_type = problem_data.get('objective', 'max')
        obj_coeffs_json = json.dumps(problem_data.get('obj_coeffs', []))
        constraints_json = json.dumps(problem_data.get('constraints', []))
        objective_val = results_data.get('objective_value')
        # Pour Simplex, results_data['solution'] contient les variables de décision
        solution_vars_json = json.dumps(results_data.get('solution')) if results_data.get('solution') is not None else None

    elif entry_type == 'graph':
        graph_input_type = problem_data.get('input_type', 'matrix') # 'matrix' ou 'list'
        # `graph_data` peut être une chaîne (liste d'arêtes) ou un dict (matrice)
        # Nous le stockerons toujours en JSON pour la cohérence.
        graph_data_json = json.dumps(problem_data.get('graph_data_raw')) # Les données brutes entrées
        graph_is_directed_int = 1 if problem_data.get('is_directed', False) else 0
        
        # Stocker les paramètres spécifiques utilisés pour les algos
        graph_params = {
            "start_node": problem_data.get("start_node"),
            "end_node": problem_data.get("end_node"),
            "source_node": problem_data.get("source_node"),
            "sink_node": problem_data.get("sink_node")
        }
        graph_params_json = json.dumps(graph_params)
        
        # Les résultats des algorithmes de graphe sont déjà un dictionnaire dans results_data
        graph_results_json = json.dumps(results_data.get('graph_algorithms_outputs')) if results_data.get('graph_algorithms_outputs') is not None else None
        # Pour les graphes, 'objective_value' et 'solution_vars' ne sont pas directement applicables de la même manière que Simplex
        # On pourrait stocker le poids du MST ou le flux max dans objective_value si on le souhaite, mais séparer est plus clair.

    try:
        db.execute(
            """
            INSERT INTO history (
                problem_type, 
                objective_type, objective_coeffs, constraints,
                graph_data_input_type, graph_data, graph_is_directed, graph_params, graph_results,
                status, objective_value, solution_vars, 
                warning
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_type,
                obj_type, obj_coeffs_json, constraints_json,
                graph_input_type, graph_data_json, graph_is_directed_int, graph_params_json, graph_results_json,
                status, objective_val, solution_vars_json,
                warning_msg
            )
        )
        db.commit()
        print(f"INFO: Entrée d'historique ({entry_type}) ajoutée à la base de données.")
    except sqlite3.Error as e:
        print(f"ERREUR BDD: Échec de l'ajout de l'entrée d'historique: {e}")
    except TypeError as e:
         print(f"ERREUR SÉRIALISATION JSON: Échec de l'ajout de l'entrée d'historique: {e}")