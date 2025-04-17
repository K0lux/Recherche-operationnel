from flask import Flask, render_template, request, flash # flash n'est pas utilisé ici, mais pourrait l'être
import numpy as np
import sys
import io # Pour capturer les impressions formatées

# Augmenter la limite de récursion (peut être nécessaire)
# sys.setrecursionlimit(2000)

# --- Copie et Adaptation de la classe SimplexSolver ---
# (Modifications principales: retourner les résultats au lieu d'imprimer)

class SimplexSolver:
    """
    Résout un problème de programmation linéaire en utilisant l'algorithme Simplex.
    Adapté pour retourner les étapes pour une application web.
    """

    def __init__(self, objective_coeffs, constraint_coeffs, rhs_constraints, objective='max', var_names=None):
        # ... (Initialisation comme avant) ...
        self.objective = objective.lower()
        self.objective_coeffs_orig = np.array(objective_coeffs, dtype=float)
        self.constraint_coeffs = np.array(constraint_coeffs, dtype=float)
        self.rhs_constraints = np.array(rhs_constraints, dtype=float)
        self.num_vars = len(objective_coeffs)
        self.num_constraints = len(rhs_constraints)
        self.warning_message = None # Pour stocker les avertissements

        if self.constraint_coeffs.shape != (self.num_constraints, self.num_vars):
            raise ValueError("Dimensions incohérentes entre les coefficients des contraintes et le nombre de variables/contraintes.")
        if len(self.rhs_constraints) != self.num_constraints:
            raise ValueError("Dimensions incohérentes entre le membre droit et le nombre de contraintes.")
        if np.any(self.rhs_constraints < 0):
             self.warning_message = ("Le membre droit (RHS) contient des valeurs négatives. "
                                     "L'algorithme Simplex standard suppose des RHS >= 0. "
                                     "Le résultat pourrait être incorrect ou l'algorithme échouer. "
                                     "(Dual Simplex / Two-Phase non implémenté)")


        if var_names and len(var_names) == self.num_vars:
            self.var_names = var_names
        else:
            self.var_names = [f'x{i+1}' for i in range(self.num_vars)]

        self.slack_var_names = [f's{i+1}' for i in range(self.num_constraints)]
        self.all_var_names = self.var_names + self.slack_var_names

        if self.objective == 'min':
            self.objective_coeffs = -self.objective_coeffs_orig
        else:
            self.objective_coeffs = self.objective_coeffs_orig

        self._build_initial_tableau()
        self.iteration_results = [] # Stocke les étapes pour l'affichage web
        self.iteration = 0
        self.max_iterations = 100

    def _build_initial_tableau(self):
        # ... (Construction du tableau comme avant) ...
        num_total_vars = self.num_vars + self.num_constraints
        self.tableau = np.zeros((self.num_constraints + 1, num_total_vars + 1))
        self.tableau[:self.num_constraints, :self.num_vars] = self.constraint_coeffs
        self.tableau[:self.num_constraints, self.num_vars:num_total_vars] = np.identity(self.num_constraints)
        self.tableau[:self.num_constraints, -1] = self.rhs_constraints
        self.tableau[-1, :self.num_vars] = -self.objective_coeffs
        self.tableau[-1, -1] = 0
        self.basic_vars_indices = list(range(self.num_vars, num_total_vars))

    def _find_pivot_column(self):
        # ... (Logique comme avant) ...
        objective_row = self.tableau[-1, :-1]
        min_val = np.min(objective_row)
        if min_val >= -1e-9:
            return -1
        else:
            return np.argmin(objective_row)

    def _find_pivot_row(self, pivot_col_index):
        # ... (Logique comme avant, mais sans les prints de debug) ...
        rhs_col = self.tableau[:-1, -1]
        pivot_col = self.tableau[:-1, pivot_col_index]
        min_ratio = float('inf')
        pivot_row_index = -1
        for i in range(self.num_constraints):
            if pivot_col[i] > 1e-9:
                ratio = rhs_col[i] / pivot_col[i]
                if ratio < min_ratio - 1e-9 :
                    min_ratio = ratio
                    pivot_row_index = i
                elif abs(ratio - min_ratio) < 1e-9:
                     pass # Garde le premier trouvé pour la simplicité

        return pivot_row_index


    def _pivot(self, pivot_row_index, pivot_col_index):
        # ... (Logique comme avant) ...
        pivot_element = self.tableau[pivot_row_index, pivot_col_index]
        if abs(pivot_element) < 1e-9:
             raise RuntimeError(f"Élément pivot ({pivot_element}) proche de zéro.")
        self.tableau[pivot_row_index, :] /= pivot_element
        for i in range(self.tableau.shape[0]):
            if i != pivot_row_index:
                multiplier = self.tableau[i, pivot_col_index]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row_index, :]
        self.basic_vars_indices[pivot_row_index] = pivot_col_index


    def _format_tableau_to_string(self, current_tableau):
        """Formate le tableau en une chaîne de caractères préformatée."""
        # Utilise StringIO pour capturer une 'impression' formatée
        output = io.StringIO()
        header = ["Base"] + self.all_var_names + ["RHS"]
        col_widths = [max(len(h), 8) for h in header] # Largeur minimale 8

        # Ligne d'en-tête
        header_line = " | ".join(f"{h:<{col_widths[j]}}" for j, h in enumerate(header))
        print(header_line, file=output)
        # Séparateur
        separator = "-+-".join("-" * width for width in col_widths)
        print(separator, file=output)

        # Lignes de contraintes
        for i in range(self.num_constraints):
            base_var_index = self.basic_vars_indices[i]
            base_var_name = self.all_var_names[base_var_index]
            row_data = [f"{base_var_name:<{col_widths[0]}}"] + \
                       [f"{val:<{col_widths[j+1]}.3f}" for j, val in enumerate(current_tableau[i, :])]
            print(" | ".join(row_data), file=output)

        # Ligne objectif (Z)
        obj_row_data = [f"{'Z':<{col_widths[0]}}"] + \
                       [f"{val:<{col_widths[j+1]}.3f}" for j, val in enumerate(current_tableau[-1, :])]
        print(separator, file=output) # Séparateur avant la ligne Z
        print(" | ".join(obj_row_data), file=output)

        return output.getvalue()

    def solve(self):
        """Exécute l'algorithme Simplex et retourne les étapes."""
        self.iteration_results = [] # Réinitialiser les résultats

        # Enregistrer le tableau initial
        initial_tableau_str = self._format_tableau_to_string(self.tableau)
        self.iteration_results.append({
            "title": "Tableau Initial",
            "tableau_str": initial_tableau_str,
            "message": f"Problème: {self.objective.capitalize()} Z"
        })

        while self.iteration < self.max_iterations:
            self.iteration += 1
            current_tableau_copy = np.round(self.tableau.copy(), 6) # Copie arrondie pour l'affichage

            pivot_col_index = self._find_pivot_column()

            # Condition d'arrêt: Optimalité
            if pivot_col_index == -1:
                final_tableau_str = self._format_tableau_to_string(current_tableau_copy)
                self.iteration_results.append({
                     "title": f"Itération {self.iteration} - Optimalité Atteinte",
                     "tableau_str": final_tableau_str,
                     "message": "Condition d'optimalité atteinte (tous Cj - Zj >= 0 dans la ligne Z)."
                 })
                solution_data = self.get_solution()
                return "optimal", solution_data, self.iteration_results

            entering_var = self.all_var_names[pivot_col_index]
            pivot_row_index = self._find_pivot_row(pivot_col_index)

            # Condition d'arrêt: Problème non borné
            if pivot_row_index == -1:
                current_tableau_str = self._format_tableau_to_string(current_tableau_copy)
                self.iteration_results.append({
                     "title": f"Itération {self.iteration} - Problème Non Borné",
                     "tableau_str": current_tableau_str,
                     "message": f"Variable entrante {entering_var}. Problème non borné détecté (aucun ratio positif).",
                     "pivot_info": {
                        "entering": entering_var, "leaving": "N/A", "pivot_val": "N/A",
                        "row_idx": "N/A", "col_idx": pivot_col_index
                     }
                 })
                return "unbounded", None, self.iteration_results

            leaving_var_global_index = self.basic_vars_indices[pivot_row_index]
            leaving_var = self.all_var_names[leaving_var_global_index]
            pivot_element = self.tableau[pivot_row_index, pivot_col_index]

            iter_message = f"Pivotage sur l'élément [{pivot_row_index}, {pivot_col_index}] = {pivot_element:.3f}"
            pivot_info = {
                 "entering": entering_var, "leaving": leaving_var, "pivot_val": f"{pivot_element:.3f}",
                 "row_idx": pivot_row_index, "col_idx": pivot_col_index
            }

            # Enregistrer l'état *avant* le pivotage pour cette itération
            tableau_before_pivot_str = self._format_tableau_to_string(current_tableau_copy)
            self.iteration_results.append({
                 "title": f"Itération {self.iteration} - Avant Pivotage",
                 "tableau_str": tableau_before_pivot_str,
                 "message": iter_message,
                 "pivot_info": pivot_info
             })


            # Effectuer le pivotage
            try:
                 self._pivot(pivot_row_index, pivot_col_index)
            except RuntimeError as e:
                 # Gérer l'erreur de pivot si elle se produit
                 self.iteration_results.append({
                    "title": f"Itération {self.iteration} - Erreur de Pivotage",
                    "tableau_str": tableau_before_pivot_str, # Montre le tableau avant l'erreur
                    "message": f"Erreur lors du pivotage: {e}",
                 })
                 return "error", None, self.iteration_results


            # Optionnel: Enregistrer aussi le tableau *après* pivotage dans la même itération ou la suivante
            # Pourrait rendre la liste d'itérations très longue. Ici, on le verra au début de l'itération suivante.

        # Condition d'arrêt: Max iterations
        final_tableau_str = self._format_tableau_to_string(self.tableau)
        self.iteration_results.append({
             "title": f"Limite d'Itérations Atteinte ({self.max_iterations})",
             "tableau_str": final_tableau_str,
             "message": "Nombre maximum d'itérations atteint. La solution peut ne pas être optimale."
         })
        solution_data = self.get_solution() # Obtenir la solution actuelle
        return "max_iterations", solution_data, self.iteration_results

    def get_solution(self):
        """Extrait la solution du tableau actuel."""
        solution = {var: 0.0 for var in self.var_names}
        slack_solution = {var: 0.0 for var in self.slack_var_names}
        objective_value = self.tableau[-1, -1]

        for i in range(self.num_constraints):
            basic_var_index = self.basic_vars_indices[i]
            value = self.tableau[i, -1]
            var_name = self.all_var_names[basic_var_index]
            if basic_var_index < self.num_vars: # Variable de décision
                solution[var_name] = value
            else: # Variable d'écart
                 slack_solution[var_name] = value


        # Ajuster la valeur de l'objectif si c'était une minimisation
        final_objective_value = -objective_value if self.objective == 'min' else objective_value

        return {
            "variables": solution,
            "slacks": slack_solution,
            "objective_value": final_objective_value
        }

# --- Flask App ---
app = Flask(__name__)
app.secret_key = 'une_cle_secrete_difficile_a_deviner' # Important pour flash messages si utilisés

@app.route('/')
def index():
    # Affiche simplement le formulaire vide
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    results_data = None
    error_message = None
    warning_message = None

    try:
        # 1. Récupérer les données du formulaire
        objective_type = request.form.get('objective_type', 'max')
        num_vars = int(request.form.get('num_vars'))
        num_constraints = int(request.form.get('num_constraints'))

        if num_vars <= 0 or num_constraints <= 0:
             raise ValueError("Le nombre de variables et de contraintes doit être positif.")

        # Coefficients Objectif
        objective_coeffs = []
        for i in range(num_vars):
            coeff_str = request.form.get(f'obj_coeff_{i}')
            if coeff_str is None: raise ValueError(f"Coefficient objectif manquant pour x{i+1}")
            objective_coeffs.append(float(coeff_str))

        # Coefficients et RHS des Contraintes
        constraint_coeffs = []
        rhs_constraints = []
        for i in range(num_constraints):
            row_coeffs = []
            for j in range(num_vars):
                coeff_str = request.form.get(f'constraint_{i}_{j}')
                if coeff_str is None: raise ValueError(f"Coefficient manquant pour la contrainte {i+1}, variable x{j+1}")
                row_coeffs.append(float(coeff_str))
            constraint_coeffs.append(row_coeffs)

            rhs_str = request.form.get(f'rhs_{i}')
            if rhs_str is None: raise ValueError(f"Membre droit (RHS) manquant pour la contrainte {i+1}")
            rhs_constraints.append(float(rhs_str))

        # 2. Créer et lancer le Solver
        var_names = [f'x{i+1}' for i in range(num_vars)] # Noms génériques
        solver = SimplexSolver(
            objective_coeffs=objective_coeffs,
            constraint_coeffs=constraint_coeffs,
            rhs_constraints=rhs_constraints,
            objective=objective_type,
            var_names=var_names
        )

        status, solution_details, iterations = solver.solve()
        warning_message = solver.warning_message # Récupérer l'avertissement potentiel de l'init

        # 3. Préparer les résultats pour le template
        status_map = {
            "optimal": "Optimale",
            "unbounded": "Non Borné",
            "max_iterations": "Limite d'itérations atteinte",
            "error": "Erreur de calcul"
        }
        status_class_map = { # Pour le style CSS
             "optimal": "success",
             "unbounded": "warning",
             "max_iterations": "warning",
             "error": "error"
        }

        results_data = {
            "status": status_map.get(status, "Inconnu"),
            "status_class": status_class_map.get(status, ""),
            "iterations": iterations,
            "objective_type": objective_type,
            "solution": None,
            "slack_solution": None,
            "objective_value": None,
        }
        if solution_details:
            results_data["solution"] = solution_details["variables"]
            results_data["slack_solution"] = solution_details["slacks"]
            results_data["objective_value"] = solution_details["objective_value"]


    except ValueError as e:
        error_message = f"Erreur de saisie ou de configuration: {e}"
    except Exception as e:
        # Capturer les erreurs inattendues du solver ou autre
        error_message = f"Une erreur interne est survenue: {e}"
        import traceback
        print("--- TRACEBACK ---")
        traceback.print_exc() # Imprime dans la console du serveur Flask pour le debug
        print("--- END TRACEBACK ---")


    # 4. Rendre le template avec les résultats ou l'erreur
    # request.form est passé pour repeupler le formulaire en cas d'erreur/résultat
    return render_template('index.html', results=results_data, error=error_message, warning=warning_message, request=request)


if __name__ == '__main__':
    # Mettre debug=False en production
    app.run(debug=True)