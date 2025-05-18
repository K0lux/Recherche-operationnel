import os
import json
from flask import Flask, render_template, request, g # g pour la BDD
from collections import defaultdict # Utile pour la reconstruction de adj pour les algos Python

# Importations locales (adaptez les chemins si votre structure est différente)
# Si db.py est au même niveau que ce fichier app.py:
from db import db_manager as db # Renommé pour clarté
# Si simplex_solver etc. sont dans un sous-dossier 'app_logic' par exemple :
# from .app_logic.simplex_solver import SimplexSolver
# from .app_logic.problem_parser import parse_problem_form
# from .app_logic import graph_algorithms
# Pour l'instant, je suppose qu'ils sont au même niveau ou dans un package 'app'
# comme dans votre structure précédente :
import datetime 
from app.simplex_solver import SimplexSolver
from app.problem_parser import parse_problem_form
from app import graph_algorithms


# --- Configuration Flask ---
DATABASE = 'simplex_history.sqlite' # Nom du fichier BDD

app = Flask(__name__)
# Si app.py est à la racine de simplex_app, et DATABASE aussi
app.config['DATABASE'] = os.path.join(app.root_path, DATABASE)
# Sinon, si DATABASE est dans un sous-dossier 'db_folder' par exemple:
# app.config['DATABASE'] = os.path.join(app.root_path, 'db_folder', DATABASE)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))

db.init_app(app) # Initialise les commandes BDD avec l'app

# --- Filtres Jinja pour un affichage amélioré dans l'historique ---
@app.template_filter('pretty_json')
def pretty_json_filter(value):
    if value is None: return "N/A"
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return str(value) # Retourner la valeur originale si ce n'est pas du JSON

@app.template_filter('pretty_constraints_simplex')
def pretty_constraints_simplex_filter(value):
    """Affiche les contraintes Simplex (formatées pour x1, x2...)."""
    if not value: return "N/A"
    try:
        constraints = json.loads(value) if isinstance(value, str) else value
        if not isinstance(constraints, list): return str(value)
        lines = []
        for c in constraints:
            if not isinstance(c, dict): continue
            coeffs = c.get("coeffs", [])
            terms = []
            for i, coeff_val in enumerate(coeffs):
                if coeff_val == 0: continue
                var = f"x{i+1}"
                if coeff_val == 1: terms.append(var)
                elif coeff_val == -1: terms.append(f"-{var}")
                else: terms.append(f"{coeff_val:g}{var}") # :g pour format compact
            lhs = " + ".join(terms).replace(" + -", " - ") if terms else "0"
            op_map = {"le": "≤", "ge": "≥", "eq": "="}
            op_display = op_map.get(c.get("type", "le"), c.get("type"))
            rhs_display = f"{c.get('rhs', 0):g}"
            lines.append(f"{lhs} {op_display} {rhs_display}")
        return "<br>".join(lines) if lines else "Aucune contrainte"
    except Exception as e:
        return f"(Erreur formatage contraintes: {e})"

@app.template_filter('pretty_objective')
def pretty_objective_filter(value, var_prefix="x"):
    """Affiche la fonction objectif."""
    if not value: return "N/A"
    try:
        coeffs = json.loads(value) if isinstance(value, str) else value
        if not isinstance(coeffs, list): return str(value)
        terms = []
        for i, coeff_val in enumerate(coeffs):
            if abs(coeff_val) < 1e-9 : continue # Ignorer coeffs nuls
            var = f"{var_prefix}{i+1}"
            term = ""
            if coeff_val == 1: term = var
            elif coeff_val == -1: term = f"-{var}"
            else: term = f"{coeff_val:g}{var}"

            if terms and coeff_val > 0: # Ajouter '+' si ce n'est pas le premier terme positif
                terms.append(f"+ {term}")
            else: # Premier terme, ou terme négatif (le signe est déjà dans 'term')
                terms.append(term)
        return " ".join(terms).replace(" + -", " - ") if terms else "0"
    except Exception as e:
        return f"(Erreur formatage objectif: {e})"

@app.template_filter('pretty_solution_vars_simplex')
def pretty_solution_vars_simplex_filter(value):
    """Affiche les variables de solution Simplex."""
    if not value: return "N/A"
    try:
        vars_dict = json.loads(value) if isinstance(value, str) else value
        if not isinstance(vars_dict, dict): return str(value)
        lines = [f"{k} = {float(v):.4f}" for k, v in vars_dict.items()]
        return "<br>".join(lines) if lines else "Aucune variable"
    except Exception as e:
        return f"(Erreur formatage solution: {e})"

# --- Routes ---
@app.route('/')
def index_route():
    return render_template('index.html', request=request)

@app.route('/history')
def history_route():
    con = db.get_db()
    raw_history_entries = con.execute(
        """SELECT id, timestamp, problem_type, 
                  objective_type, objective_coeffs, constraints,
                  graph_data_input_type, graph_data, graph_is_directed, graph_params, graph_results,
                  status, objective_value, solution_vars, 
                  warning
           FROM history ORDER BY timestamp DESC"""
    ).fetchall()

    import json
    history_entries_processed = []
    for entry_row in raw_history_entries:
        entry_dict = dict(entry_row) # Convertir sqlite3.Row en dictionnaire modifiable
        try:
            # Tenter de parser la chaîne timestamp en objet datetime
            timestamp_str = entry_dict['timestamp']
            dt_obj = None
            possible_formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"]
            for fmt in possible_formats:
                try:
                    dt_obj = datetime.datetime.strptime(timestamp_str, fmt)
                    break
                except ValueError:
                    continue
            if dt_obj:
                entry_dict['timestamp'] = dt_obj
            else:
                print(f"AVERTISSEMENT: Impossible de parser le timestamp '{timestamp_str}' pour l'entrée ID {entry_dict['id']}. Utilisation de la chaîne brute.")
        except Exception as e:
            print(f"Erreur lors du traitement du timestamp pour l'entrée ID {entry_dict['id']}: {e}")
        # Désérialiser les champs JSON si besoin
        for key in ['solution_vars', 'graph_results', 'objective_coeffs', 'constraints', 'graph_data', 'graph_params']:
            if key in entry_dict and isinstance(entry_dict[key], str):
                try:
                    entry_dict[key] = json.loads(entry_dict[key])
                except Exception:
                    pass  # Laisse la valeur brute si ce n'est pas du JSON
        history_entries_processed.append(entry_dict)

    return render_template('history.html', history_entries=history_entries_processed)

@app.route('/solve', methods=['POST'])
def solve_route():
    results_dict = {}
    error_message = None
    warning_message_from_solver = None
    problem_data_for_db = {} 

    try:
        problem_data, parse_error = parse_problem_form(request.form)
        if parse_error:
            error_message = f"Erreur de saisie: {parse_error}"
        else:
            problem_data_for_db = problem_data.copy()
            solver = SimplexSolver(problem_data) # Assurez-vous que SimplexSolver est bien importé
            status, solution_details, iterations, duality_data, sensitivity_data, interpretation_data = solver.solve()
            warning_message_from_solver = solver.warning_message

            status_map = {
                solver.STATUS_OPTIMAL: "Optimale", solver.STATUS_UNBOUNDED: "Non Borné",
                solver.STATUS_MAX_ITERATIONS: "Limite d'itérations atteinte",
                solver.STATUS_ERROR: "Erreur de calcul", solver.STATUS_INFEASIBLE: "Infaisable"
            }
            status_class_map = {
                solver.STATUS_OPTIMAL: "success", solver.STATUS_UNBOUNDED: "warning",
                solver.STATUS_MAX_ITERATIONS: "warning", solver.STATUS_ERROR: "error",
                solver.STATUS_INFEASIBLE: "error"
            }
            results_dict = {
                "status": status_map.get(status, "Inconnu"),
                "status_class": status_class_map.get(status, ""),
                "iterations": iterations, "objective_type": problem_data['objective'],
                "standard_form_str": solver.get_standard_form_string(),
                "needs_artificial_vars": solver.needs_phase1,
                "variable_types": solver.variable_types, "solution": None,
                "slack_surplus_solution": None, "objective_value": None,
                "duality": duality_data, "sensitivity": sensitivity_data,
                "interpretation": interpretation_data, "warning_message": warning_message_from_solver
            }
            if solution_details:
                results_dict["solution"] = solution_details["variables"]
                results_dict["slack_surplus_solution"] = solution_details["slack_surplus"]
                results_dict["objective_value"] = solution_details["objective_value"]

            if status != solver.STATUS_ERROR and not parse_error:
                 db.add_history_entry('simplex', problem_data_for_db, results_dict)
            else:
                 print("INFO: Erreur Simplex ou parsing, entrée non ajoutée à l'historique.")
    except ValueError as e: 
        if not error_message: error_message = str(e)
    except Exception as e:
        error_message = f"Erreur interne inattendue: {e}"
        import traceback
        traceback.print_exc()
    return render_template('index.html', results=results_dict, error=error_message,
                           warning=warning_message_from_solver, request=request)

@app.route('/graphs', methods=['GET', 'POST'])
def graphs_route():
    graph_input_form_data = {} 
    graph_results_display = {} # Initialiser à un dict vide pour éviter UndefinedError
    graph_error_display = None
    graph_data_for_js_viz = None

    if request.method == 'GET':
        graph_input_form_data['num_nodes'] = request.args.get('num_nodes', 3)
        graph_input_form_data['is_directed'] = request.args.get('is_directed') == 'true'
        # Conserver les autres paramètres pour le repeuplement
        for param in ['start_node', 'end_node', 'source_node', 'sink_node']:
            graph_input_form_data[param] = request.args.get(param, '')

    if request.method == 'POST':
        graph_input_form_data = request.form.to_dict() # Pour repeuplement
        problem_data_for_db_graphs = {}
        try:
            num_nodes = int(graph_input_form_data.get('num_nodes', 0))
            is_directed_form = graph_input_form_data.get('is_directed') == 'true'
            
            problem_data_for_db_graphs['is_directed'] = is_directed_form
            problem_data_for_db_graphs['input_type'] = 'matrix'

            node_names_list = []
            if num_nodes > 0:
                for i in range(num_nodes):
                    name = graph_input_form_data.get(f'node_name_{i}', '').strip()
                    if not name: name = chr(65 + i) 
                    if name in node_names_list:
                        raise ValueError(f"Nom de nœud '{name}' dupliqué.")
                    node_names_list.append(name)
            problem_data_for_db_graphs['node_names'] = node_names_list

            adj_matrix_form_data = {} # Pour graph_algorithms.parse_adjacency_matrix
            raw_matrix_for_db = {}    # Pour la BDD (avec indices i-j)
            if num_nodes > 0:
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        val_str = graph_input_form_data.get(f'adj_matrix_{i}_{j}', "")
                        adj_matrix_form_data[f"{i}-{j}"] = val_str
                        raw_matrix_for_db[f"{i}-{j}"] = val_str 
            problem_data_for_db_graphs['graph_data_raw'] = raw_matrix_for_db

            # Récupérer les paramètres des algorithmes
            start_node = graph_input_form_data.get('start_node', '').strip()
            end_node = graph_input_form_data.get('end_node', '').strip()
            source_node = graph_input_form_data.get('source_node', '').strip()
            sink_node = graph_input_form_data.get('sink_node', '').strip()

            problem_data_for_db_graphs.update({
                'start_node': start_node, 'end_node': end_node,
                'source_node': source_node, 'sink_node': sink_node
            })
            
            if num_nodes <= 0:
                graph_error_display = "Le nombre de nœuds doit être positif."
            else:
                adj, nodes_set, edges_for_mst_and_viz = graph_algorithms.parse_adjacency_matrix(
                    num_nodes, node_names_list, adj_matrix_form_data, is_directed=is_directed_form
                )
                
                vis_nodes = list(nodes_set)
                vis_edges = [{"from": u, "to": v, "weight": w} for u, v, w in edges_for_mst_and_viz]
                graph_data_for_js_viz = {
                    "nodes": vis_nodes, "edges": vis_edges,
                    "is_directed": is_directed_form, "dijkstra_path_edges": []
                }
                
                # --- Exécution des Algorithmes ---
                graph_results_display = {}

                # Dijkstra
                graph_results_display['dijkstra'] = {}
                if start_node: # Exécuter seulement si un nœud de départ est fourni
                    try:
                        if start_node not in nodes_set: raise ValueError(f"Nœud de départ Dijkstra '{start_node}' inconnu.")
                        target_node = end_node if end_node and end_node in nodes_set else None
                        if end_node and not target_node: raise ValueError(f"Nœud d'arrivée Dijkstra '{end_node}' inconnu.")

                        dist, path, _, _ = graph_algorithms.dijkstra(adj, start_node, target_node)
                        if target_node:
                            if dist == float('inf'):
                                graph_results_display['dijkstra']['error'] = f"Aucun chemin de {start_node} à {target_node}."
                            else:
                                graph_results_display['dijkstra']['path'] = path
                                graph_results_display['dijkstra']['distance'] = dist
                                if path and len(path) > 1:
                                    for k_path in range(len(path) - 1):
                                        # Trouver poids pour l'arête du chemin (simplifié)
                                        edge_w = next((w for v_n, w in adj.get(path[k_path],[]) if v_n == path[k_path+1]), 0)
                                        graph_data_for_js_viz["dijkstra_path_edges"].append({"from": path[k_path], "to": path[k_path+1], "weight": edge_w})
                        else:
                            graph_results_display['dijkstra']['message'] = f"Distances calculées depuis {start_node}."
                    except ValueError as e_dij: graph_results_display['dijkstra']['error'] = str(e_dij)
                    except Exception as e_dij_gen: graph_results_display['dijkstra']['error'] = f"Erreur Dijkstra: {e_dij_gen}"
                else:
                     graph_results_display['dijkstra']['error'] = "Nœud de départ non spécifié."


                # Ford-Fulkerson
                graph_results_display['ford_fulkerson'] = {}
                if source_node and sink_node:
                    try:
                        if source_node not in nodes_set: raise ValueError(f"Nœud source F-F '{source_node}' inconnu.")
                        if sink_node not in nodes_set: raise ValueError(f"Nœud puits F-F '{sink_node}' inconnu.")
                        # F-F a besoin d'une liste d'adjacence où les poids sont des capacités.
                        # `adj` devrait déjà être correct si `is_directed_form` a été utilisé pour le parsing.
                        max_flow = graph_algorithms.ford_fulkerson(adj, nodes_set, source_node, sink_node)
                        graph_results_display['ford_fulkerson']['max_flow'] = max_flow
                    except ValueError as e_ff: graph_results_display['ford_fulkerson']['error'] = str(e_ff)
                    except Exception as e_ff_gen: graph_results_display['ford_fulkerson']['error'] = f"Erreur F-F: {e_ff_gen}"
                else:
                    graph_results_display['ford_fulkerson']['error'] = "Source et puits non spécifiés."

                # Prim (traite le graphe comme non-dirigé)
                graph_results_display['prim'] = {}
                prim_start_node_param = start_node if start_node in nodes_set else None
                try:
                    adj_undirected_for_prim = defaultdict(list)
                    temp_nodes_for_prim = set()
                    for u_mst, v_mst, w_mst in edges_for_mst_and_viz: # Utiliser la liste d'arêtes unique
                        adj_undirected_for_prim[u_mst].append((v_mst, w_mst))
                        adj_undirected_for_prim[v_mst].append((u_mst, w_mst))
                        temp_nodes_for_prim.add(u_mst); temp_nodes_for_prim.add(v_mst)
                    
                    if not temp_nodes_for_prim and num_nodes > 0 : temp_nodes_for_prim = nodes_set # Si pas d'arêtes mais des noeuds

                    if not temp_nodes_for_prim: # Graphe vide
                         graph_results_display['prim'] = {"total_weight": 0, "mst_edges": []}
                    else:
                        prim_weight, prim_edges = graph_algorithms.prim_mst(adj_undirected_for_prim, temp_nodes_for_prim, prim_start_node_param)
                        graph_results_display['prim'] = {"total_weight": prim_weight, "mst_edges": prim_edges}
                except Exception as e_prim: graph_results_display['prim']['error'] = f"Erreur Prim: {str(e_prim)}"

                # Kruskal (traite le graphe comme non-dirigé)
                graph_results_display['kruskal'] = {}
                try:
                    if not nodes_set: # Graphe vide
                        graph_results_display['kruskal'] = {"total_weight": 0, "mst_edges": []}
                    else:
                        kruskal_weight, kruskal_edges = graph_algorithms.kruskal_mst(nodes_set, edges_for_mst_and_viz)
                        graph_results_display['kruskal'] = {"total_weight": kruskal_weight, "mst_edges": kruskal_edges}
                except Exception as e_kruskal: graph_results_display['kruskal']['error'] = f"Erreur Kruskal: {str(e_kruskal)}"
                
                # --- Fin Exécution Algorithmes ---

                results_for_db_graphs = {
                    "status": "Terminé", 
                    "graph_algorithms_outputs": graph_results_display,
                    "warning_message": None 
                }
                db.add_history_entry('graph', problem_data_for_db_graphs, results_for_db_graphs)
        
        except ValueError as e:
            graph_error_display = str(e)
            graph_data_for_js_viz = None 
        except Exception as e:
            graph_error_display = f"Une erreur inattendue est survenue: {str(e)}"
            graph_data_for_js_viz = None
            import traceback
            traceback.print_exc()

    return render_template('graphs.html', 
                           graph_input_form=graph_input_form_data,
                           graph_results=graph_results_display, 
                           graph_error=graph_error_display,
                           graph_data_for_js=graph_data_for_js_viz,
                           request=request)

# --- if __name__ == '__main__': (comme avant) ---
if __name__ == '__main__':
    db_path = os.path.join(app.root_path, DATABASE) # Utiliser app.root_path pour la BDD
    if not os.path.exists(db_path):
         with app.app_context():
              print(f"INFO: BDD non trouvée à {db_path}. Initialisation...")
              db.init_db()
              print("INFO: BDD initialisée.")
    else:
         print(f"INFO: BDD trouvée à {db_path}.")
    app.run(debug=True, host='0.0.0.0') # host='0.0.0.0' pour accès réseau local