# simplex_app/simplex_solver.py

import numpy as np
import io
from math import inf, isinf # Pour l'infini mathématique et sa vérification

class SimplexSolver:
    """
    Résout un problème de Programmation Linéaire (PL) en utilisant la Méthode des Deux Phases
    si nécessaire. Gère les contraintes <=, >=, et =.
    Effectue des analyses post-optimalité (Dualité, Sensibilité, Interprétation)
    si une solution optimale finie est trouvée.
    """
    # États possibles du solveur
    STATUS_OPTIMAL = "Optimale"
    STATUS_UNBOUNDED = "Non Borné"
    STATUS_INFEASIBLE = "Infaisable"
    STATUS_MAX_ITERATIONS = "Limite d'itérations atteinte"
    STATUS_ERROR = "Erreur de calcul"
    TOLERANCE = 1e-9 # Tolérance pour les comparaisons avec zéro

    def __init__(self, problem_data):
        """
        Initialisation du solveur avec les données du problème.
        problem_data: dict attendu de problem_parser.parse_form_data
        """
        self.problem_data = problem_data
        self.objective = problem_data['objective']
        self.objective_coeffs_orig = np.array(problem_data['obj_coeffs'], dtype=float)
        self.num_vars_decision = len(self.objective_coeffs_orig)
        self.num_constraints = len(problem_data['constraints'])
        self.var_names_decision = problem_data['var_names']

        # Initialisation des attributs
        self.warning_message = None
        self.needs_phase1 = False
        self.is_solvable_by_simple_solver = False # Sera déterminé dans _prepare_standard_form
        self.variable_types = {} # e.g., {'x1': 'Decision', 's1': 'Slack', ...}
        self.constraint_map = {} # e.g., {'s1': 0} (slack s1 vient de la contrainte originale 0)
        
        self.slack_vars = []
        self.surplus_vars = []
        self.artificial_vars = []
        self.artificial_var_indices = [] # Indices globaux des variables artificielles

        self.all_var_names_ordered = [] # Liste de tous les noms de variables dans l'ordre standard
        self.constraint_coeffs_standard = np.array([])
        self.rhs_standard = np.array([])
        self.initial_basis_indices = [] # Indices globaux des variables initialement en base

        # --- Préparation ---
        self._prepare_standard_form() # Analyse les contraintes, définit les variables, etc.

        # --- État pour la résolution ---
        self.tableau = None # Tableau Simplex NumPy
        self.tableau_var_names = [] # Noms des variables DANS le tableau actuel (peut changer entre phases)
        self.basic_vars_indices = [] # Indices GLOBaux des variables actuellement en base
        
        self.phase = 1 if self.needs_phase1 else 2 # Détermine la phase de démarrage
        self.iteration = 0 # Compteur global d'itérations
        self.max_iterations = 100 # Sécurité
        self.iteration_results = [] # Log des tableaux/étapes
        
        self.status = None # Statut final de la résolution
        self.initial_tableau_built = False

        # Variables pour stocker les résultats finaux de l'analyse
        self.final_tableau = None # Copie du dernier tableau
        self.final_solution_details = None
        self.final_duality_data = None
        self.final_sensitivity_data = None
        self.final_interpretation_data = None

    def _prepare_standard_form(self):
        """
        Transforme le problème original en forme standard Ax=b, x>=0.
        Gère les RHS négatifs, ajoute les variables d'écart (slack), d'excédent (surplus)
        et artificielles. Détermine si la Phase 1 est nécessaire.
        """
        self.slack_vars = []
        self.surplus_vars = []
        self.artificial_vars = []
        processed_coeffs_list = [] # Liste temporaire pour les coefficients des contraintes
        processed_rhs_list = []    # Liste temporaire pour les membres droits
        self.initial_basis_indices_map = {} # map index_contrainte_originale -> index_global_var_base_initiale

        var_idx_counter = self.num_vars_decision # Compteur pour les indices globaux des nouvelles variables
        for name in self.var_names_decision:
            self.variable_types[name] = 'Decision'

        # Étape 1: Parcourir les contraintes, gérer RHS, identifier types et vars extra
        for i, constraint in enumerate(self.problem_data['constraints']):
            coeffs = np.array(constraint['coeffs'], dtype=float)
            rhs = float(constraint['rhs'])
            ctype = constraint['type']

            # Assurer RHS >= 0
            if rhs < -self.TOLERANCE: # Utiliser tolérance pour la comparaison
                rhs = -rhs
                coeffs = -coeffs # Inverser tous les coefficients de la ligne
                if ctype == 'le': ctype = 'ge'
                elif ctype == 'ge': ctype = 'le'
                # 'eq' reste 'eq', mais avec ligne inversée
                if self.warning_message is None: self.warning_message = ""
                self.warning_message += f"Contrainte C{i+1} inversée (RHS d'origine < 0). "
            elif abs(rhs) < self.TOLERANCE: # Si RHS est numériquement zéro
                 rhs = 0.0

            processed_rhs_list.append(rhs)
            current_row_coeffs_decision_part = list(coeffs) # Coeffs des variables de décision pour cette contrainte

            base_var_index_for_this_row = -1 # Index global de la variable initialement en base pour cette contrainte
            extra_coeffs_info = [] # Liste de tuples: (coefficient, nom_variable, type_variable, est_artificielle)

            if ctype == 'le':
                slack_name = f's{len(self.slack_vars) + 1}'
                self.slack_vars.append(slack_name)
                self.variable_types[slack_name] = 'Ecart (Slack)'
                self.constraint_map[slack_name] = i # s_i liée à contrainte originale i
                extra_coeffs_info.append((1.0, slack_name, 'Slack', False))
                base_var_index_for_this_row = var_idx_counter # Slack est en base
                var_idx_counter += 1
            elif ctype == 'ge':
                self.needs_phase1 = True # Une contrainte >= nécessite la Phase 1
                surplus_name = f'e{len(self.surplus_vars) + 1}'
                artif_name = f'a{len(self.artificial_vars) + 1}'
                self.surplus_vars.append(surplus_name)
                self.artificial_vars.append(artif_name)
                self.variable_types[surplus_name] = 'Excedent (Surplus)'
                self.variable_types[artif_name] = 'Artificielle'
                self.constraint_map[surplus_name] = i
                self.constraint_map[artif_name] = i # Artificielle aussi liée à cette contrainte
                extra_coeffs_info.append((-1.0, surplus_name, 'Surplus', False)) # -e_i
                extra_coeffs_info.append((1.0, artif_name, 'Artificial', True))  # +a_i
                # L'index global de l'artificielle sera var_idx_counter + 1 (car surplus ajouté en premier)
                self.artificial_var_indices.append(var_idx_counter + 1)
                base_var_index_for_this_row = var_idx_counter + 1 # Artificielle est en base
                var_idx_counter += 2 # Une var surplus, une var artificielle
            elif ctype == 'eq':
                self.needs_phase1 = True # Une contrainte = nécessite la Phase 1
                artif_name = f'a{len(self.artificial_vars) + 1}'
                self.artificial_vars.append(artif_name)
                self.variable_types[artif_name] = 'Artificielle'
                self.constraint_map[artif_name] = i
                extra_coeffs_info.append((1.0, artif_name, 'Artificial', True)) # +a_i
                self.artificial_var_indices.append(var_idx_counter)
                base_var_index_for_this_row = var_idx_counter # Artificielle est en base
                var_idx_counter += 1
            
            processed_coeffs_list.append({
                'decision': current_row_coeffs_decision_part, # Coeffs des x_j
                'extras': extra_coeffs_info                   # Infos sur s_i, e_i, a_i
            })
            self.initial_basis_indices_map[i] = base_var_index_for_this_row

        # Étape 2: Construire les noms ordonnés et la matrice standard complète
        self.num_vars_slack = len(self.slack_vars)
        self.num_vars_surplus = len(self.surplus_vars)
        self.num_vars_artificial = len(self.artificial_vars)
        self.num_vars_total = self.num_vars_decision + self.num_vars_slack + self.num_vars_surplus + self.num_vars_artificial

        self.all_var_names_ordered = (
            list(self.var_names_decision) + 
            self.slack_vars + 
            self.surplus_vars + 
            self.artificial_vars
        )
        # Créer un mapping nom -> index global pour faciliter le remplissage de la matrice
        var_name_to_global_index = {name: idx for idx, name in enumerate(self.all_var_names_ordered)}

        self.constraint_coeffs_standard = np.zeros((self.num_constraints, self.num_vars_total))
        self.rhs_standard = np.array(processed_rhs_list, dtype=float)

        for i in range(self.num_constraints):
            data = processed_coeffs_list[i]
            # Coeffs des variables de décision
            self.constraint_coeffs_standard[i, :self.num_vars_decision] = data['decision']
            # Coeffs des variables extra (slack/surplus/artificielle)
            for coeff_val, var_name, _, _ in data['extras']:
                global_idx = var_name_to_global_index[var_name]
                self.constraint_coeffs_standard[i, global_idx] = coeff_val
        
        # Étape 3: Définir la liste des indices globaux des variables initialement en base
        self.initial_basis_indices = [self.initial_basis_indices_map[i] for i in range(self.num_constraints)]

        # Étape 4: Déterminer si le solveur simple peut être utilisé (après que tout soit défini)
        self.is_solvable_by_simple_solver = False # Par défaut
        if not self.needs_phase1: # Si pas besoin de Phase 1 (pas de contraintes >= ou =)
            # Toutes les variables en base initiale doivent être des slacks
            all_initial_base_are_slacks = True
            for base_idx in self.initial_basis_indices:
                var_name_in_base = self.all_var_names_ordered[base_idx]
                if self.variable_types.get(var_name_in_base) != 'Ecart (Slack)':
                    all_initial_base_are_slacks = False
                    break
            
            if all_initial_base_are_slacks and np.all(self.rhs_standard >= -self.TOLERANCE):
                self.is_solvable_by_simple_solver = True
            elif np.any(self.rhs_standard < -self.TOLERANCE): # Même si pas de P1, RHS neg -> pas simple
                if self.warning_message is None: self.warning_message = ""
                self.warning_message += "Contient RHS négatif après transformation, solveur simple non applicable sans Dual Simplex. "
        
        # Log informatif
        if self.needs_phase1:
            print("INFO: Problème nécessite la Méthode des Deux Phases.")
        elif self.is_solvable_by_simple_solver:
            print("INFO: Problème peut être résolu directement (cas simple tout <=, RHS >=0).")
        else:
            # Ce cas peut arriver si, par exemple, on a un RHS négatif qui a été inversé,
            # transformant une contrainte <= en >=, nécessitant alors Phase 1 (déjà couvert),
            # ou si un autre pré-traitement complexe serait nécessaire.
            # Le warning sur RHS négatif est déjà géré plus haut.
            print("INFO: Problème ne nécessite pas Phase 1 (pas de >= ou =) mais n'est pas un cas simple direct pour initialisation Phase 2.")

    # ... (Collez TOUTES les autres méthodes ici : _build_phase1_tableau, _drive_out_artificials, etc.
    #      en vous assurant que l'indentation est correcte et qu'elles sont membres de la classe SimplexSolver)
    #      Je vais les réintégrer ici pour la complétude.

       # >>>>> MÉTHODE log_iteration À AJOUTER/COMPLÉTER <<<<<
    def log_iteration(self, title, tableau, message="", pivot_info=None, ratios=None, min_ratio_index=None):
        """Enregistre les détails d'une itération pour affichage."""
        if tableau is None: # Sécurité
            tableau_str = "Tableau non disponible pour cette itération."
        else:
            try:
                # S'assurer que _format_tableau_to_string est bien défini dans cette classe
                tableau_str = self._format_tableau_to_string(tableau, ratios, min_ratio_index, phase=self.phase)
            except AttributeError as e:
                 print(f"ERREUR dans log_iteration: _format_tableau_to_string manquant? {e}")
                 tableau_str = "Erreur lors du formatage du tableau (méthode manquante)."
            except Exception as e:
                print(f"ERREUR lors du formatage du tableau pour log_iteration ('{title}'): {e}")
                tableau_str = "Erreur lors du formatage du tableau."
        
        self.iteration_results.append({
            "title": title,
            "tableau_str": tableau_str,
            "message": message,
            "pivot_info": pivot_info
        })

    # >>>>> MÉTHODE get_standard_form_string À AJOUTER/COMPLÉTER <<<<<
    def get_standard_form_string(self):
        """Retourne une représentation textuelle de la forme standard du problème."""
        # S'assurer que self.constraint_coeffs_standard et self.rhs_standard sont initialisés
        if not hasattr(self, 'constraint_coeffs_standard') or not hasattr(self, 'rhs_standard'):
            return "Forme standard non encore préparée."
        if self.constraint_coeffs_standard.size == 0 or self.rhs_standard.size == 0:
             return "Données pour la forme standard non disponibles."


        output = io.StringIO()

        # Objectif (basé sur les coefficients originaux)
        obj_str = f"{self.objective.capitalize()} Z = "
        coeffs_parts = []
        for i, (coeff_val, var_name) in enumerate(zip(self.objective_coeffs_orig, self.var_names_decision)):
            # Gérer l'affichage de +g pour les nombres (signe si non premier, pas de signe + si premier)
            # et g pour éviter les zéros inutiles après la virgule.
            term = f"{coeff_val:g}{var_name}"
            if i > 0 and coeff_val >= 0: # Ajouter + pour les termes suivants positifs
                term = f"+ {term}"
            elif coeff_val < 0: # Gérer l'espace pour les négatifs
                 term = f"- {abs(coeff_val):g}{var_name}"

            if i == 0 and coeff_val < 0: # Premier terme négatif
                 term = f"-{abs(coeff_val):g}{var_name}"
            elif i == 0 and coeff_val >=0: # Premier terme positif (pas de +)
                 term = f"{coeff_val:g}{var_name}"


            coeffs_parts.append(term)
        
        obj_str += " ".join(coeffs_parts) if coeffs_parts else "0"
        # Nettoyage final des doubles signes (ex: "+ -5x" -> "- 5x")
        obj_str = obj_str.replace('+ -', '- ')
        print(obj_str, file=output)
        print("Sujet à:", file=output)

        # Contraintes de la forme standard (Ax = b)
        for i in range(self.num_constraints):
            constraint_parts = []
            first_term = True
            for j in range(self.num_vars_total): # Utiliser toutes les variables de la forme standard
                coeff = self.constraint_coeffs_standard[i, j]
                if abs(coeff) > self.TOLERANCE: # Si le coefficient est significatif
                    var_name = self.all_var_names_ordered[j]
                    val_str = f"{abs(coeff):g}" if abs(coeff) != 1 else "" # Ne pas afficher "1"x_i
                    
                    term_sign = ""
                    if not first_term: # Pas le premier terme de la contrainte
                        term_sign = "+ " if coeff > 0 else "- "
                    else: # Premier terme
                        term_sign = "- " if coeff < 0 else "" # Pas de "+" pour le premier terme positif
                    
                    constraint_parts.append(f"{term_sign}{val_str}{var_name}")
                    first_term = False

            if not constraint_parts: constraint_parts.append("0") # Si tous les coeffs sont nuls
            rhs_val = self.rhs_standard[i]
            # Utiliser join puis replace pour gérer les espaces autour des opérateurs
            print(f"  {' '.join(constraint_parts).replace(' + -', ' - ').replace('  ', ' ')} = {rhs_val:g}", file=output)

        # Non-négativité pour toutes les variables
        non_neg_vars = ", ".join(self.all_var_names_ordered)
        print(f"  {non_neg_vars} >= 0", file=output)

        return output.getvalue()
    
    def _build_phase1_tableau(self):
        """Construit le tableau Simplex initial pour la Phase 1 (minimiser W = sum a_i)."""
        print("INFO: Construction du tableau Phase 1.")
        num_vars_tableau = self.num_vars_total # Toutes les variables (décision, s, e, a)
        self.tableau_var_names = list(self.all_var_names_ordered) # Noms pour ce tableau

        self.tableau = np.zeros((self.num_constraints + 1, num_vars_tableau + 1))
        # Remplir coefficients des contraintes et RHS
        self.tableau[:self.num_constraints, :num_vars_tableau] = self.constraint_coeffs_standard
        self.tableau[:self.num_constraints, -1] = self.rhs_standard

        # Objectif W: Maximiser -W = -Sum a_i.
        # Ligne initiale (avant mise en forme canonique): -W + sum(a_i) = 0
        w_row = np.zeros(num_vars_tableau + 1) # Ligne pour -W
        for global_artif_idx in self.artificial_var_indices:
            w_row[global_artif_idx] = 1.0 # Coeff +1 pour chaque a_i dans l'expression de -W

        # Mettre la ligne W en forme canonique:
        # Pour chaque ligne 'k' où une variable artificielle a_k (ou a_j) est en base,
        # on doit soustraire la ligne k du tableau de la ligne W pour annuler le coeff de a_k.
        for k in range(self.num_constraints):
            # self.initial_basis_indices[k] est l'index GLOBAL de la variable en base à la ligne k
            if self.initial_basis_indices[k] in self.artificial_var_indices:
                 w_row -= self.tableau[k, :] # Soustraire la ligne k entière

        self.tableau[-1, :] = w_row # Ligne W (ou -W) formatée
        self.basic_vars_indices = list(self.initial_basis_indices) # Base initiale de Phase 1
        self.initial_tableau_built = True

    def _drive_out_artificials(self):
        """
        Après Phase 1 (W=0), tente de pivoter les variables artificielles (à valeur 0)
        hors de la base, si possible, en utilisant des variables non artificielles.
        Retourne True si des modifications ont été faites, False sinon.
        """
        print("INFO: Tentative de suppression des variables artificielles de la base (Phase 1 terminée, W=0).")
        made_change = False
        
        # Itérer sur une copie des indices de base car self.basic_vars_indices peut changer
        current_basic_vars = list(self.basic_vars_indices) 

        for r in range(self.num_constraints): # r est l'index de la ligne du tableau
            basis_idx_global = current_basic_vars[r] # Index GLOBAL de la var en base à la ligne r

            if basis_idx_global in self.artificial_var_indices:
                 # Une variable artificielle est en base. Sa valeur (RHS[r]) doit être 0.
                 if abs(self.tableau[r, -1]) > self.TOLERANCE:
                     print(f"AVERTISSEMENT: Artificielle {self.all_var_names_ordered[basis_idx_global]} "
                           f"en base (ligne {r}) avec valeur non nulle {self.tableau[r, -1]:.3g} "
                           f"bien que Min W = 0. Cela indique un problème. On ne pivote pas.")
                     continue 

                 print(f"INFO: Artificielle {self.all_var_names_ordered[basis_idx_global]} "
                       f"en base (ligne {r}, valeur 0). Recherche d'un pivot non-artificiel...")
                 
                 pivot_col_found_global = -1
                 # Chercher un pivot potentiel dans la ligne r, parmi les variables NON artificielles
                 # Les variables non artificielles sont: décision, slack, surplus
                 num_vars_non_artificial = self.num_vars_decision + self.num_vars_slack + self.num_vars_surplus
                 
                 for c_global in range(num_vars_non_artificial): # c_global est l'index GLOBAL
                     # Vérifier si l'élément du tableau est non nul
                     if abs(self.tableau[r, c_global]) > self.TOLERANCE:
                         # Il faut aussi s'assurer que cette variable c_global n'est pas déjà
                         # en base DANS UNE AUTRE LIGNE. Si c'est le cas, la pivoter ici
                         # rendrait la base linéairement dépendante (ce qui est ok si ligne redondante).
                         # Pour la simplicité, on prend le premier pivot non-artificiel trouvé.
                         pivot_col_found_global = c_global
                         break 
                 
                 if pivot_col_found_global != -1:
                     entering_var_name = self.all_var_names_ordered[pivot_col_found_global]
                     leaving_var_name = self.all_var_names_ordered[basis_idx_global]
                     print(f"INFO: Pivotage pour sortir artificielle: Ligne={r}, Colonne (globale)={pivot_col_found_global}. "
                           f"Entrante={entering_var_name}, Sortante={leaving_var_name}.")
                     try:
                         # _pivot attend l'index de colonne du TABLEAU actuel.
                         # Puisque le tableau de Phase 1 contient toutes les colonnes,
                         # l'index global EST l'index de colonne du tableau.
                         self._pivot(r, pivot_col_found_global) # Met à jour self.basic_vars_indices
                         made_change = True
                         # Mettre à jour current_basic_vars pour refléter le changement pour les itérations suivantes
                         current_basic_vars = list(self.basic_vars_indices)
                     except RuntimeError as e:
                         print(f"ERREUR: Échec du pivotage pour sortir l'artificielle "
                               f"{leaving_var_name} (ligne {r}): {e}")
                 else:
                     # Aucun pivot non-artificiel trouvé. La ligne est soit redondante,
                     # soit toutes les variables non-artificielles ont un coeff nul dans cette ligne.
                     # Dans ce cas, la variable artificielle reste (à 0). Elle sera ignorée en Phase 2.
                     artif_var_name = self.all_var_names_ordered[basis_idx_global]
                     print(f"INFO: Impossible de pivoter l'artificielle {artif_var_name} (ligne {r}) "
                           f"hors de la base. La ligne peut être redondante ou ne contenir que des zéros "
                           f"pour les variables non-artificielles.")
        return made_change

    def _build_phase2_tableau(self, final_phase1_tableau):
        """
        Prépare le tableau pour la Phase 2 à partir du tableau final de la Phase 1.
        Supprime les colonnes des variables artificielles et calcule la nouvelle ligne Z.
        """
        print("INFO: Préparation du tableau Phase 2.")
        tableau_p1 = final_phase1_tableau # Tableau complet de fin de Phase 1
        num_vars_p1_total = tableau_p1.shape[1] - 1 # Nombre total de colonnes de variables dans P1

        # 1. Calculer la ligne Z (objectif original) en fonction de la base FINALE de Phase 1
        #    Cette base peut encore contenir des artificielles (à valeur 0).
        z_row_full_length = np.zeros(num_vars_p1_total + 1) # +1 pour RHS
        
        # Coeffs originaux pour Maximisation (soit Max Z, soit Max -Z si Min original)
        original_obj_coeffs_for_maximization = self.objective_coeffs_orig if self.objective == 'max' else -self.objective_coeffs_orig
        
        # Mettre les -Cj dans la ligne Z pour les variables de décision
        z_row_full_length[:self.num_vars_decision] = -original_obj_coeffs_for_maximization
        # Les coeffs des slacks et surplus sont 0 dans l'objectif original, donc -0 = 0.
        # Les coeffs des artificielles sont aussi considérés comme 0 pour l'objectif Z.

        # Ajuster la ligne Z pour qu'elle soit en forme canonique par rapport à la base actuelle
        # (qui est self.basic_vars_indices, mise à jour par _drive_out_artificials potentiellement)
        for k_row_idx in range(self.num_constraints): # Pour chaque ligne k du tableau
            basic_var_global_index = self.basic_vars_indices[k_row_idx] # Indice GLOBAL de la var en base
            
            cb_coeff = 0.0 # Coefficient de la variable de base dans l'objectif original (pour maximisation)
            if basic_var_global_index < self.num_vars_decision: # Si c'est une variable de décision
                 cb_coeff = original_obj_coeffs_for_maximization[basic_var_global_index]
            # Si c'est un slack ou surplus, son cb_coeff est 0.
            # Si c'est une artificielle (ne devrait pas arriver avec valeur > 0), cb_coeff est 0.
            
            if abs(cb_coeff) > self.TOLERANCE: # Si le coeff de la var de base dans Z est non nul
                 # Z_row = Z_row - cb_coeff * (ligne_k_du_tableau_P1)
                 # Mais comme Z_row contient -Cj, on fait: Z_row = Z_row + C_B_k * ligne_k
                 # ou si on veut Zj-Cj:  (Cb B^-1 A_j - Cj)
                 # Ici, on part de -Cj, donc on ajoute Cb * (ligne pivotée k) pour éliminer Cb.
                 z_row_full_length += cb_coeff * tableau_p1[k_row_idx, :]


        # 2. Identifier les colonnes à garder (non artificielles + colonne RHS)
        #    et les noms de variables correspondants pour le tableau de Phase 2.
        cols_to_keep_indices = []
        phase2_var_names = []
        for global_idx, var_name in enumerate(self.all_var_names_ordered):
            if var_name not in self.artificial_vars: # Garder si pas artificielle
                cols_to_keep_indices.append(global_idx)
                phase2_var_names.append(var_name)
        
        # Ajouter l'index de la colonne RHS (qui est le dernier)
        rhs_col_index_in_p1 = num_vars_p1_total 
        cols_to_keep_indices.append(rhs_col_index_in_p1)

        # 3. Construire le tableau Phase 2 réduit (sans les colonnes artificielles)
        # Prendre les lignes de contraintes du tableau_p1 et les colonnes sélectionnées
        tableau_p2_constraint_rows = tableau_p1[:-1, cols_to_keep_indices] # Exclure la ligne W (objectif Phase 1)
        
        # Prendre les éléments correspondants de la ligne Z calculée
        z_row_phase2 = z_row_full_length[cols_to_keep_indices]

        self.tableau = np.vstack([tableau_p2_constraint_rows, z_row_phase2])
        
        # 4. Mettre à jour les noms de variables pour le tableau de Phase 2
        self.tableau_var_names = phase2_var_names
        # self.basic_vars_indices (indices globaux) reste correct.
        
        self.initial_tableau_built = True # Le tableau est prêt pour les itérations de Phase 2
        print("INFO: Tableau Phase 2 construit et colonnes artificielles (si existantes) supprimées.")

    def _try_build_initial_tableau_for_simple_solver(self):
        """
        Construit directement le tableau pour la Phase 2 si le problème est "simple"
        (tout <=, RHS >= 0, pas besoin de Phase 1).
        """
        if not self.is_solvable_by_simple_solver: # Vérification de sécurité
             self.initial_tableau_built = False
             self.warning_message = (self.warning_message or "") + \
                                    "Tentative de construction de tableau simple sur problème non simple."
             print("ERREUR: _try_build_initial_tableau_for_simple_solver appelé sur un problème non simple.")
             return

        print("INFO: Construction directe du tableau Phase 2 (cas simple: tout <=, RHS >= 0).")
        # Le tableau ne contiendra que les variables de décision et les slacks
        self.num_vars_in_tableau_simple = self.num_vars_decision + self.num_vars_slack
        self.tableau_var_names = list(self.var_names_decision) + self.slack_vars # Noms pour ce tableau

        self.tableau = np.zeros((self.num_constraints + 1, self.num_vars_in_tableau_simple + 1))
        
        # Coefficients des variables de décision (issus des contraintes <= originales)
        self.tableau[:self.num_constraints, :self.num_vars_decision] = \
            self.constraint_coeffs_standard[:self.num_constraints, :self.num_vars_decision]
        
        # Variables d'écart (doivent former une matrice identité ici)
        # Les indices globaux des slacks dans constraint_coeffs_standard
        slack_global_indices_in_standard_matrix = [
            idx for idx, name in enumerate(self.all_var_names_ordered) 
            if name in self.slack_vars
        ]
        # Les colonnes des slacks dans constraint_coeffs_standard
        source_slack_columns = self.constraint_coeffs_standard[:self.num_constraints, slack_global_indices_in_standard_matrix]
        
        # La destination dans le tableau simple (après les variables de décision)
        destination_slack_columns_in_tableau = self.tableau[:self.num_constraints, self.num_vars_decision : self.num_vars_in_tableau_simple]

        if source_slack_columns.shape == destination_slack_columns_in_tableau.shape:
            self.tableau[:self.num_constraints, self.num_vars_decision : self.num_vars_in_tableau_simple] = source_slack_columns
        else:
            # Cela ne devrait pas arriver si _prepare_standard_form est correct pour ce cas
            print(f"AVERTISSEMENT: Discordance de forme pour les slacks en cas simple. "
                  f"Source: {source_slack_columns.shape}, Dest: {destination_slack_columns_in_tableau.shape}")
            # Tentative de copie partielle (peut masquer un problème plus profond)
            min_cols = min(source_slack_columns.shape[1], destination_slack_columns_in_tableau.shape[1])
            self.tableau[:self.num_constraints, self.num_vars_decision : self.num_vars_decision + min_cols] = source_slack_columns[:, :min_cols]


        # Membres droits (RHS)
        self.tableau[:self.num_constraints, -1] = self.rhs_standard # RHS des contraintes originales (ajustés si <0)

        # Ligne Objectif (Z)
        # Coeffs pour Maximisation (Max Z ou Max -Z si Min original)
        obj_coeffs_for_maximization = self.objective_coeffs_orig if self.objective == 'max' else -self.objective_coeffs_orig
        self.tableau[-1, :self.num_vars_decision] = -obj_coeffs_for_maximization # -Cj pour les vars de décision
        # Les slacks ont un coût nul, donc Zj-Cj = 0 - 0 = 0 dans la ligne Z
        self.tableau[-1, self.num_vars_decision : self.num_vars_in_tableau_simple] = 0 
        self.tableau[-1, -1] = 0 # Valeur Z initiale = 0

        # Variables de base initiales (indices globaux des slacks)
        self.basic_vars_indices = list(self.initial_basis_indices) # Devrait contenir les indices globaux des slacks
        self.initial_tableau_built = True
        print("INFO: Tableau simplexe initial (cas simple) construit.")

    # ... (Le reste des méthodes comme solve, log_iteration, _run_simplex_iterations, _find_pivot_column,
    #      _find_pivot_row, _pivot, _format_tableau_to_string, get_solution, _formulate_dual,
    #      _analyze_final_tableau, _finalize_results doivent être collées ici, en s'assurant
    #      de leur indentation correcte au sein de la classe SimplexSolver.)
    #      Il est crucial que ces méthodes soient présentes et correctement indentées.
    #      Par exemple, la méthode solve orchestrera l'appel à _build_phase1_tableau ou
    #      _try_build_initial_tableau_for_simple_solver.

    # (Je vais remettre la méthode solve et _run_simplex_iterations pour la structure)

    def solve(self):
        """Orchestre la résolution complète du problème."""
        try:
            if self.needs_phase1:
                self.phase = 1
                self._build_phase1_tableau()
                if not self.initial_tableau_built: # Erreur construction P1
                    self.status = self.STATUS_ERROR
                    self.iteration_results.append({"title": "Erreur Initialisation Phase 1", "tableau_str": self.warning_message or "Impossible de construire le tableau de Phase 1.", "message":""})
                    self._finalize_results() # Pour calculer dualité au moins
                    return self.status, self.final_solution_details, self.iteration_results, self.final_duality_data, self.final_sensitivity_data, self.final_interpretation_data

                self.log_iteration("Phase 1 - Tableau Initial (Min W)", self.tableau)
                
                phase1_status = self._run_simplex_iterations() # Lance les itérations de Phase 1

                final_w_value_in_tableau = self.tableau[-1, -1] # Rappel: c'est la valeur de -W
                min_w_objective = -final_w_value_in_tableau

                if min_w_objective > self.TOLERANCE: # Si Min W > 0 (strictement)
                    self.status = self.STATUS_INFEASIBLE
                    self.log_iteration(f"Phase 1 - Terminé ({self.status})", self.tableau, 
                                       message=f"Problème original infaisable (Min W = {min_w_objective:.4g} > 0).")
                    self._finalize_results()
                    return self.status, self.final_solution_details, self.iteration_results, self.final_duality_data, self.final_sensitivity_data, self.final_interpretation_data
                else: # Min W = 0 (ou très proche de 0)
                    self.log_iteration("Phase 1 - Terminé (Min W ≈ 0)", self.tableau, message="Base réalisable trouvée. Tentative de nettoyage de la base...")
                    self._drive_out_artificials() # Essayer de pivoter les artificielles hors de la base
                    # Logguer le tableau APRÈS nettoyage potentiel des artificielles
                    self.log_iteration("Phase 1 - Tableau après nettoyage des artificielles", self.tableau)
                    
                    self._build_phase2_tableau(self.tableau) # Prépare P2 en supprimant cols artif.
                    if not self.initial_tableau_built: # Erreur construction P2
                        self.status = self.STATUS_ERROR
                        self.iteration_results.append({"title": "Erreur Initialisation Phase 2", "tableau_str": self.warning_message or "Impossible de construire le tableau de Phase 2.", "message":""})
                        self._finalize_results()
                        return self.status, self.final_solution_details, self.iteration_results, self.final_duality_data, self.final_sensitivity_data, self.final_interpretation_data
                    self.phase = 2 # Passer à la Phase 2
            
            # --- Phase 2 (ou résolution directe si Phase 1 non nécessaire) ---
            if self.phase == 2: # S'exécute si needs_phase1 était false, ou si Phase 1 a réussi
                if not self.initial_tableau_built: # Si on arrive ici sans tableau (P1 non req mais pas simple)
                    if self.is_solvable_by_simple_solver:
                        self._try_build_initial_tableau_for_simple_solver()
                    else:
                        # Ce cas signifie que P1 n'était pas nécessaire (pas d'artificielles à l'origine)
                        # MAIS le problème n'est pas "simple" pour une initialisation directe P2
                        # (ex: RHS négatifs après transformation, ou base initiale non purement slack).
                        self.initial_tableau_built = False 
                        self.warning_message = (self.warning_message or "") + \
                                               "Problème non directement initialisable sans Phase 1 complexe et non 'simple'. Phase 1 aurait dû être déclenchée si des artificielles étaient nécessaires."
                        print("INFO: Problème non 'simple' pour initialisation directe Phase 2, et Phase 1 non déclenchée.")
                
                if not self.initial_tableau_built: # Vérifier à nouveau après _try_build
                    self.status = self.STATUS_ERROR if not self.warning_message or "RHS négatif" not in self.warning_message else self.STATUS_INFEASIBLE
                    self.iteration_results.append({
                        "title": "Initialisation Phase 2 Échouée",
                        "tableau_str": self.warning_message or "Impossible de construire le tableau initial pour Phase 2.",
                        "message": "Le problème n'est pas directement soluble par le solveur actuel sans Phase 1."
                    })
                    self._finalize_results()
                    return self.status, self.final_solution_details, self.iteration_results, self.final_duality_data, self.final_sensitivity_data, self.final_interpretation_data

                # Logguer le tableau initial de Phase 2
                title_p2 = "Phase 2 - Tableau Initial (Opt Z)" if self.needs_phase1 else "Tableau Initial (Opt Z)"
                self.log_iteration(title_p2, self.tableau, message=f"Optimisation de: {self.objective.capitalize()} Z.")
                
                phase2_status = self._run_simplex_iterations() # Lance les itérations de Phase 2
                self.status = phase2_status

        except Exception as e:
            print(f"ERREUR CRITIQUE: Exception inattendue pendant la méthode solve(): {e}")
            import traceback
            traceback.print_exc()
            self.status = self.STATUS_ERROR
            self.iteration_results.append({"title": "Erreur Critique Inattendue", 
                                           "tableau_str": traceback.format_exc(), 
                                           "message": str(e)})

        # Calculs finaux et organisation des résultats (solution, dual, sensibilité)
        self._finalize_results() # Stocke les résultats dans les attributs self.final_*

        # Retourner les résultats stockés
        return (self.status, self.final_solution_details, self.iteration_results, 
                self.final_duality_data, self.final_sensitivity_data, self.final_interpretation_data)

    # (Collez ici les autres méthodes : log_iteration, _run_simplex_iterations, _find_pivot_column,
    #  _find_pivot_row, _pivot, _format_tableau_to_string, get_solution, _formulate_dual,
    #  _analyze_final_tableau, _finalize_results)
    # ...
    # Exemple pour _run_simplex_iterations (à compléter avec les autres)
    def _run_simplex_iterations(self):
        """ Boucle principale des itérations Simplex pour la phase courante. """
        current_phase = self.phase
        # Le compteur d'itération `self.iteration` est global et incrémenté dans _pivot
        # On peut utiliser un compteur local si on veut savoir combien d'itérations par phase
        local_iter_count_for_phase = 0 

        while self.iteration < self.max_iterations:
            if self.tableau is None: # Sécurité
                 print(f"ERREUR: Tableau est None en début d'itération Phase {current_phase}.")
                 return self.STATUS_ERROR
            
            tableau_before_pivot = self.tableau.copy()
            local_iter_count_for_phase += 1

            # 1. Trouver colonne pivot
            pivot_col_index_in_tableau = self._find_pivot_column() # Index relatif au tableau actuel

            # --- Optimalité Atteinte pour cette phase ---
            if pivot_col_index_in_tableau == -1:
                print(f"INFO: Phase {current_phase} - Optimalité atteinte à l'itération de phase {local_iter_count_for_phase} (globale {self.iteration}).")
                # Le tableau final sera loggué par _finalize_results si nécessaire
                return self.STATUS_OPTIMAL 

            # 2. Trouver ligne pivot
            entering_var_name = self.tableau_var_names[pivot_col_index_in_tableau]
            pivot_row_index, ratios, min_ratio = self._find_pivot_row(pivot_col_index_in_tableau)

            # --- Problème Non Borné (pour cette colonne pivot) ---
            if pivot_row_index == -1:
                print(f"INFO: Phase {current_phase} - Problème non borné détecté.")
                self.log_iteration(
                    f"Phase {current_phase} - Itération {local_iter_count_for_phase} - Non Borné",
                    tableau_before_pivot, ratios=ratios, # Afficher avec les ratios calculés
                    message=f"Variable entrante {entering_var_name}. Problème non borné détecté (aucun ratio positif ou tous éléments <= 0 dans colonne pivot).",
                    pivot_info={"entering": entering_var_name, "leaving": "N/A", "pivot_val": "N/A",
                                "row_idx": "N/A", "col_idx": pivot_col_index_in_tableau, "min_ratio": None}
                )
                # Un problème non borné en Phase 1 est une erreur (W devrait être borné par 0)
                # Un problème non borné en Phase 2 signifie que le problème original est non borné
                return self.STATUS_UNBOUNDED if current_phase == 2 else self.STATUS_ERROR

            # 3. Préparer informations pour le log et pivoter
            leaving_var_global_idx = self.basic_vars_indices[pivot_row_index] # Index global de la sortante
            leaving_var_name = self.all_var_names_ordered[leaving_var_global_idx]
            pivot_element = self.tableau[pivot_row_index, pivot_col_index_in_tableau]
            
            pivot_info = {
                 "entering": entering_var_name, "leaving": leaving_var_name, 
                 "pivot_val": f"{pivot_element:.3f}",
                 "row_idx": pivot_row_index, "col_idx": pivot_col_index_in_tableau, # Index relatifs au tableau
                 "min_ratio": min_ratio
            }
            self.log_iteration(
                 f"Phase {current_phase} - Itération {local_iter_count_for_phase} - Sélection Pivot",
                 tableau_before_pivot, # Tableau avant pivotage
                 ratios=ratios, min_ratio_index=pivot_row_index, # Avec ratios
                 message=f"Pivot sur élément [{pivot_row_index}, {pivot_col_index_in_tableau}] = {pivot_element:.3f}",
                 pivot_info=pivot_info
            )

            # --- Pivotage ---
            try:
                 # _pivot attend l'index de colonne du TABLEAU actuel.
                 self._pivot(pivot_row_index, pivot_col_index_in_tableau) 
                 # self.iteration est incrémenté globalement dans _pivot (ou juste après)
            except RuntimeError as e:
                 self.log_iteration(
                    f"Phase {current_phase} - Itération {local_iter_count_for_phase} - Erreur Pivotage",
                    tableau_before_pivot, ratios=ratios, min_ratio_index=pivot_row_index,
                    message=f"Erreur: {e}"
                 )
                 return self.STATUS_ERROR # Erreur de calcul
            
            # Le tableau après pivotage sera loggué au début de la prochaine itération
            # ou par _finalize_results si c'est la dernière.

        # --- Limite d'itérations globale atteinte ---
        print(f"INFO: Phase {current_phase} - Limite d'itérations ({self.max_iterations}) atteinte.")
        return self.STATUS_MAX_ITERATIONS

    # Assurez-vous de copier ici les versions COMPLÈTES et CORRECTEMENT INDENTÉES de:
    # _find_pivot_column, _find_pivot_row, _pivot, _format_tableau_to_string, get_solution, 
    # _formulate_dual, _analyze_final_tableau, _finalize_results, et log_iteration (déjà fait)
    # ... (Collez ces méthodes ici)
    def _find_pivot_column(self):
        """Trouve colonne pivot basée sur la phase."""
        if self.tableau is None: return -1
        objective_row = self.tableau[-1, :-1]
        num_vars_in_current_tableau = len(self.tableau_var_names)
        potential_cols = []
        if self.phase == 1:
            positive_coeffs_indices = np.where(objective_row[:num_vars_in_current_tableau] > self.TOLERANCE)[0]
            if len(positive_coeffs_indices) == 0: return -1
            potential_cols = positive_coeffs_indices
        else: # Phase 2
            negative_coeffs_indices = np.where(objective_row[:num_vars_in_current_tableau] < -self.TOLERANCE)[0]
            if len(negative_coeffs_indices) == 0: return -1
            potential_cols = negative_coeffs_indices
        if not potential_cols.any(): return -1
        return np.min(potential_cols) # Règle de Bland

    def _find_pivot_row(self, pivot_col_index):
        """Trouve ligne pivot (variable sortante) via min ratio test + Bland."""
        if self.tableau is None: return -1, [], float('inf')
        rhs_col = self.tableau[:-1, -1]
        pivot_col = self.tableau[:-1, pivot_col_index]
        min_ratio = float('inf')
        pivot_row_index = -1
        ratios = [None] * self.num_constraints
        eligible_rows_indices = []
        positive_pivot_indices = np.where(pivot_col > self.TOLERANCE)[0]
        if len(positive_pivot_indices) == 0:
            return -1, ratios, min_ratio # Non borné pour cette colonne
        for i in positive_pivot_indices:
            if abs(rhs_col[i]) < self.TOLERANCE: ratio = 0.0 # Traiter RHS=0 comme ratio=0 pour éviter division par petit pivot_col[i]
            else: ratio = rhs_col[i] / pivot_col[i]
            ratios[i] = ratio
            if abs(ratio - min_ratio) < self.TOLERANCE: eligible_rows_indices.append(i)
            elif ratio < min_ratio:
                min_ratio = ratio; pivot_row_index = i; eligible_rows_indices = [i]
        if len(eligible_rows_indices) > 1: # Règle de Bland pour départager
            min_leaving_var_idx = float('inf'); best_row_idx = -1
            for row_idx in eligible_rows_indices:
                 leaving_var_global_idx = self.basic_vars_indices[row_idx]
                 if leaving_var_global_idx < min_leaving_var_idx:
                     min_leaving_var_idx = leaving_var_global_idx; best_row_idx = row_idx
            pivot_row_index = best_row_idx
        elif len(eligible_rows_indices) == 1: pivot_row_index = eligible_rows_indices[0]
        return pivot_row_index, ratios, min_ratio

    def _pivot(self, pivot_row_index, pivot_col_index_in_tableau):
        """Effectue le pivotage de Gauss-Jordan."""
        if self.tableau is None: raise RuntimeError("Tableau non initialisé pour pivotage.")
        pivot_element = self.tableau[pivot_row_index, pivot_col_index_in_tableau]
        if abs(pivot_element) < self.TOLERANCE:
             raise RuntimeError(f"Élément pivot [{pivot_row_index},{pivot_col_index_in_tableau}] = {pivot_element} est trop proche de zéro.")
        self.tableau[pivot_row_index, :] /= pivot_element # Normaliser ligne pivot
        num_rows_in_tableau = self.tableau.shape[0]
        for i in range(num_rows_in_tableau):
            if i != pivot_row_index:
                multiplier = self.tableau[i, pivot_col_index_in_tableau]
                if abs(multiplier) > self.TOLERANCE: # Éviter opérations inutiles
                    self.tableau[i, :] -= multiplier * self.tableau[pivot_row_index, :]
        # Mettre à jour la variable de base pour cette ligne
        # pivot_col_index_in_tableau est l'index dans le TABLEAU ACTUEL (P1 ou P2 réduit)
        # Il faut trouver l'index GLOBAL de la variable entrante
        entering_var_name = self.tableau_var_names[pivot_col_index_in_tableau]
        entering_var_global_index = self.all_var_names_ordered.index(entering_var_name)
        self.basic_vars_indices[pivot_row_index] = entering_var_global_index
        self.iteration +=1 # Incrémenter le compteur global d'itérations

    # Dans votre fichier simplex_solver.py, à l'intérieur de la classe SimplexSolver

    def _format_tableau_to_string(self, current_tableau, ratios=None, min_ratio_index=None, phase=None):
        """Formate le tableau NumPy en une chaîne de caractères HTML <table>."""
        if current_tableau is None:
            return "<p>Tableau non disponible.</p>"

        output = io.StringIO()
        obj_label = "W" if phase == 1 else "Z"
        
        # Utiliser self.tableau_var_names qui contient les noms pour le tableau actuel (P1 ou P2)
        # S'assurer que self.tableau_var_names est bien défini avant cette méthode
        if not self.tableau_var_names:
             # Fallback si tableau_var_names n'est pas prêt (ne devrait pas arriver en usage normal)
             num_cols_vars = current_tableau.shape[1] - 1
             current_var_names_for_header = [f"V{j+1}" for j in range(num_cols_vars)]
             print("AVERTISSEMENT: self.tableau_var_names non défini, utilisation de noms génériques pour le tableau.")
        else:
             # Nombre de colonnes de variables dans le tableau actuel (exclut RHS)
             num_cols_vars = current_tableau.shape[1] - 1
             # S'assurer de ne pas dépasser la longueur de tableau_var_names
             current_var_names_for_header = self.tableau_var_names[:num_cols_vars]


        header_cells = ["Base (Vb)"] + current_var_names_for_header + ["RHS (b)"]
        if ratios is not None:
            header_cells.append("Ratio (θ)")

        output.write('<div class="table-responsive"><table class="simplex-table-generated table table-sm table-bordered table-hover">\n') # Ajout de classes Bootstrap pour test

        # --- En-tête du tableau ---
        output.write("  <thead>\n    <tr>\n")
        for cell_content in header_cells:
            output.write(f"      <th>{cell_content}</th>\n")
        output.write("    </tr>\n  </thead>\n")

        # --- Corps du tableau ---
        output.write("  <tbody>\n")

        # Lignes de contraintes (variables de base)
        num_constraint_rows = self.num_constraints 
        for i in range(num_constraint_rows):
            output.write("    <tr>\n")
            # Colonne Base
            base_var_name = "N/A" # Cas par défaut
            if i < len(self.basic_vars_indices): # Sécurité pour éviter IndexError
                 base_var_global_idx = self.basic_vars_indices[i]
                 if base_var_global_idx < len(self.all_var_names_ordered): # Sécurité
                     base_var_name = self.all_var_names_ordered[base_var_global_idx]
                 else:
                     print(f"AVERT: Indice de base {base_var_global_idx} hors limites pour all_var_names_ordered.")
            else:
                 print(f"AVERT: Indice de ligne {i} hors limites pour basic_vars_indices.")

            output.write(f'      <td class="base-col">{base_var_name}</td>\n')

            # Colonnes des variables (jusqu'au nombre de variables dans current_var_names_for_header)
            for j in range(len(current_var_names_for_header)):
                output.write(f"      <td>{current_tableau[i, j]:.3f}</td>\n")
            
            # Colonne RHS
            output.write(f'      <td class="rhs-col">{current_tableau[i, -1]:.3f}</td>\n')

            # Colonne Ratio
            if ratios is not None:
                ratio_val_str = ""
                if i < len(ratios) and ratios[i] is not None:
                    if isinf(ratios[i]):
                        ratio_val_str = "∞" # Symbole infini HTML
                    else:
                        ratio_val_str = f"{ratios[i]:.3f}"
                
                ratio_class = "ratio-col"
                if i == min_ratio_index:
                    ratio_class += " min-ratio-highlight" # Classe pour surligner
                output.write(f'      <td class="{ratio_class}">{ratio_val_str}</td>\n')
            output.write("    </tr>\n")

        # Ligne Objectif (W ou Z)
        output.write('    <tr class="objective-row">\n') # Classe pour la ligne objectif
        output.write(f'      <td class="base-col objective-label">{obj_label}</td>\n') # Label W ou Z

        # Valeurs de la ligne objectif pour les variables
        for j in range(len(current_var_names_for_header)):
            output.write(f"      <td>{current_tableau[-1, j]:.3f}</td>\n")
        
        # Valeur de l'objectif (coin inférieur droit)
        output.write(f'      <td class="rhs-col objective-value">{current_tableau[-1, -1]:.3f}</td>\n')

        if ratios is not None:
            output.write("      <td></td>\n") # Cellule vide pour la colonne ratio dans la ligne objectif
        output.write("    </tr>\n")

        output.write("  </tbody>\n")
        output.write("</table></div>\n") # Fermeture de table-responsive

        return output.getvalue()

    def get_solution(self):
        # (Version complète copiée depuis la réponse précédente)
        if self.tableau is None or self.status not in [self.STATUS_OPTIMAL, self.STATUS_MAX_ITERATIONS]: return None
        solution = {var: 0.0 for var in self.var_names_decision}; slack_surplus_solution = {var: 0.0 for var in self.slack_vars + self.surplus_vars}
        objective_value_in_tableau = self.tableau[-1, -1]; num_rows_tableau = self.num_constraints; rhs_column_tableau = self.tableau[:num_rows_tableau, -1]
        for i_row_solution in range(num_rows_tableau):
             if i_row_solution < len(self.basic_vars_indices):
                 basic_var_global_index_sol = self.basic_vars_indices[i_row_solution]; var_name_sol = self.all_var_names_ordered[basic_var_global_index_sol]
                 value_sol = max(0.0, rhs_column_tableau[i_row_solution]) # Assurer non-négativité
                 if var_name_sol in solution: solution[var_name_sol] = value_sol
                 elif var_name_sol in slack_surplus_solution: slack_surplus_solution[var_name_sol] = value_sol
        final_objective_value_calc = None
        if self.phase == 2: # Seulement si la Phase 2 a été atteinte (ou était la seule phase)
            if self.objective == 'min': final_objective_value_calc = -objective_value_in_tableau
            else: final_objective_value_calc = objective_value_in_tableau
        elif self.status == self.STATUS_MAX_ITERATIONS and self.phase == 1: final_objective_value_calc = None
        elif self.status == self.STATUS_INFEASIBLE: final_objective_value_calc = None
        return {"variables": solution, "slack_surplus": slack_surplus_solution, "objective_value": final_objective_value_calc}

    def _formulate_dual(self):
        # (Version complète copiée depuis la réponse précédente)
        primal = self.problem_data; num_primal_constraints = len(primal['constraints']); num_primal_vars = len(primal['obj_coeffs'])
        dual_objective_type = 'min' if primal['objective'] == 'max' else 'max'; dual_var_prefix = 'y'
        dual_vars = [f"{dual_var_prefix}{i+1}" for i in range(num_primal_constraints)]; output = io.StringIO()
        dual_obj_coeffs = [c['rhs'] for c in primal['constraints']]; obj_str = f"{dual_objective_type.capitalize()} W = "; coeffs_parts = []
        for i, coeff in enumerate(dual_obj_coeffs):
             sign = '+' if coeff >= 0 else '-'; val = abs(coeff)
             if not coeffs_parts and sign == '+': sign = ''; coeffs_parts.append(f"{sign} {val:g}{dual_vars[i]}")
        obj_str += " ".join(coeffs_parts).replace('+ -', '- ') if coeffs_parts else "0"; print(obj_str, file=output); print("Sujet à:", file=output)
        primal_A = np.array([c['coeffs'] for c in primal['constraints']]); A_T = primal_A.T
        for j in range(num_primal_vars):
            constraint_parts = []
            for i in range(num_primal_constraints):
                coeff = A_T[j, i]
                if abs(coeff) > self.TOLERANCE:
                     sign = '+' if coeff >= 0 else '-'; val = abs(coeff)
                     if not constraint_parts and sign == '+': sign = ''; constraint_parts.append(f"{sign} {val:g}{dual_vars[i]}")
            if not constraint_parts: constraint_parts.append("0")
            dual_constraint_sign = '>=' if primal['objective'] == 'max' else '<='; rhs_val = primal['obj_coeffs'][j]
            print(f"  {' '.join(constraint_parts).replace('+ -', '- ')} {dual_constraint_sign} {rhs_val:g}", file=output)
        dual_var_signs = []
        for i, constraint in enumerate(primal['constraints']):
             sign = ""; ctype = constraint['type']
             if ctype == 'le': sign = '>=' if primal['objective'] == 'max' else '<='
             elif ctype == 'ge': sign = '<=' if primal['objective'] == 'max' else '>='
             elif ctype == 'eq': sign = 'URS'
             if sign != 'URS': dual_var_signs.append(f"{dual_vars[i]} {sign} 0")
             else: dual_var_signs.append(f"{dual_vars[i]} {sign}")
        print(f"  {', '.join(dual_var_signs)}", file=output)
        return {"formulation": output.getvalue()}

    def _analyze_final_tableau(self, allow_non_optimal=False):
        # (Version complète copiée depuis la réponse précédente, avec les adaptations pour P2)
        if self.phase != 2: return None, None;
        if self.tableau is None: return None, None
        print("INFO: Analyse post-optimale (Sensibilité/Interprétation)...")
        sensitivity = {"objective_ranges": [], "rhs_ranges": []}; interpretation = {"reduced_costs": {}, "shadow_prices": {}}
        final_tableau = self.tableau; final_obj_row = final_tableau[-1, :]; num_tableau_rows = self.num_constraints; B_inv = None
        slack_indices_in_p2_tableau = []
        current_p2_col_idx = 0
        for name in self.tableau_var_names:
             if name in self.slack_vars: slack_indices_in_p2_tableau.append(current_p2_col_idx)
             current_p2_col_idx +=1
        if len(slack_indices_in_p2_tableau) == self.num_constraints:
             try: B_inv = final_tableau[:num_tableau_rows, slack_indices_in_p2_tableau]
             except IndexError as e: print(f"ERREUR B_inv: {e}"); B_inv = None
        else: B_inv = None; print(f"AVERT: Nombre slacks ({len(slack_indices_in_p2_tableau)}) != contraintes ({self.num_constraints}) pour B_inv.")
        current_var_idx_map = {name: idx for idx, name in enumerate(self.tableau_var_names)}
        for j in range(self.num_vars_decision):
            var_name = self.var_names_decision[j]; global_idx = self.all_var_names_ordered.index(var_name)
            if global_idx not in self.basic_vars_indices:
                if var_name in current_var_idx_map:
                    tableau_col_idx = current_var_idx_map[var_name]; cj_minus_zj_val = final_obj_row[tableau_col_idx]
                    reduced_cost = -cj_minus_zj_val if self.objective == 'max' else cj_minus_zj_val
                    interpretation["reduced_costs"][var_name] = reduced_cost
        for i in range(self.num_constraints):
             slack_var_name = None
             for s_name in self.slack_vars:
                  if self.constraint_map.get(s_name) == i: slack_var_name = s_name; break
             if slack_var_name and slack_var_name in current_var_idx_map:
                  tableau_col_idx = current_var_idx_map[slack_var_name]; zj_minus_cj_slack = final_obj_row[tableau_col_idx]
                  shadow_price = zj_minus_cj_slack if self.objective == 'max' else -zj_minus_cj_slack
                  interpretation["shadow_prices"][slack_var_name] = shadow_price
        if B_inv is None: return sensitivity, interpretation
        for j in range(self.num_vars_decision):
            var_name = self.var_names_decision[j]; original_coeff = self.objective_coeffs_orig[j]
            decrease, increase = float('inf'), float('inf'); global_idx = self.all_var_names_ordered.index(var_name)
            if global_idx in self.basic_vars_indices:
                 try:
                      row_index_in_tableau = self.basic_vars_indices.index(global_idx)
                      for k_p2, var_k_name in enumerate(self.tableau_var_names):
                           global_idx_k = self.all_var_names_ordered.index(var_k_name)
                           if global_idx_k not in self.basic_vars_indices:
                                a_lk = final_tableau[row_index_in_tableau, k_p2]; zj_minus_cj_k = final_obj_row[k_p2]
                                if abs(a_lk) > self.TOLERANCE:
                                     ratio = -zj_minus_cj_k / a_lk
                                     if a_lk > 0: decrease = min(decrease, ratio)
                                     else: increase = min(increase, -ratio)
                 except Exception as e: print(f"AVERT Cj (base) {var_name}: {e}")
            else:
                if var_name in current_var_idx_map:
                    tableau_col_idx = current_var_idx_map[var_name]; zj_minus_cj_val = final_obj_row[tableau_col_idx]
                    if self.objective == 'max': increase = zj_minus_cj_val; decrease = float('inf')
                    else: decrease = zj_minus_cj_val; increase = float('inf')
            decrease_str = f"{decrease:.3f}" if not isinf(decrease) else "+inf"; increase_str = f"{increase:.3f}" if not isinf(increase) else "+inf"
            lower = original_coeff - decrease if not isinf(decrease) else "-inf"; upper = original_coeff + increase if not isinf(increase) else "+inf"
            lower_str = f"{lower:.3f}" if isinstance(lower, float) else lower; upper_str = f"{upper:.3f}" if isinstance(upper, float) else upper
            sensitivity["objective_ranges"].append({ "var": var_name, "original": original_coeff, "decrease": decrease_str, "increase": increase_str, "lower_bound": lower_str, "upper_bound": upper_str})
        rhs_final = final_tableau[:num_tableau_rows, -1]
        for i in range(self.num_constraints):
             slack_var_name_for_rhs = f's{i+1}' # Suppose que s_i correspond à la contrainte originale i
             original_rhs_val = self.problem_data['constraints'][i]['rhs']
             shadow_price_for_rhs = interpretation["shadow_prices"].get(slack_var_name_for_rhs, 0.0)
             decrease_rhs, increase_rhs = float('inf'), float('inf')
             if B_inv is not None and i < B_inv.shape[1]: # S'assurer que l'index est valide pour B_inv
                 try:
                     b_inv_col_for_rhs = B_inv[:, i]
                     for k_row_rhs in range(num_tableau_rows):
                          b_inv_val_for_rhs = b_inv_col_for_rhs[k_row_rhs]
                          if abs(b_inv_val_for_rhs) > self.TOLERANCE:
                               ratio_rhs = rhs_final[k_row_rhs] / b_inv_val_for_rhs
                               if b_inv_val_for_rhs > 0: decrease_rhs = min(decrease_rhs, ratio_rhs)
                               else: increase_rhs = min(increase_rhs, -ratio_rhs)
                 except Exception as e: print(f"AVERT RHS C{i+1}: {e}"); decrease_rhs, increase_rhs = 0.0, 0.0
             else: # Si B_inv non dispo ou index invalide
                  decrease_rhs, increase_rhs = 0.0, 0.0 # Indiquer intervalle [original, original]
                  if B_inv is None: print(f"INFO: B_inv non disponible pour sensibilité RHS C{i+1}")
                  else: print(f"INFO: Index de contrainte {i} hors limites pour B_inv (shape {B_inv.shape}) pour sensibilité RHS.")

             decrease_rhs_str = f"{decrease_rhs:.3f}" if not isinf(decrease_rhs) else "+inf"; increase_rhs_str = f"{increase_rhs:.3f}" if not isinf(increase_rhs) else "+inf"
             current_rhs_in_solver_val = self.rhs_standard[i]
             lower_rhs = current_rhs_in_solver_val - decrease_rhs if not isinf(decrease_rhs) else "-inf"; upper_rhs = current_rhs_in_solver_val + increase_rhs if not isinf(increase_rhs) else "+inf"
             lower_rhs_str = f"{lower_rhs:.3f}" if isinstance(lower_rhs, float) else lower_rhs; upper_rhs_str = f"{upper_rhs:.3f}" if isinstance(upper_rhs, float) else upper_rhs
             sensitivity["rhs_ranges"].append({ "constraint_index": i, "var": slack_var_name_for_rhs, "original": original_rhs_val, "decrease": decrease_rhs_str, "increase": increase_rhs_str, "lower_bound": lower_rhs_str, "upper_bound": upper_rhs_str, "shadow_price": shadow_price_for_rhs})
        return sensitivity, interpretation

    def _finalize_results(self):
        # (Version complète copiée depuis la réponse précédente)
        try: self.final_duality_data = self._formulate_dual()
        except Exception as e: print(f"ERREUR dualité: {e}"); self.final_duality_data = None
        if self.status == self.STATUS_OPTIMAL or self.status == self.STATUS_MAX_ITERATIONS:
            try: self.final_solution_details = self.get_solution()
            except Exception as e: print(f"ERREUR solution: {e}"); self.final_solution_details = None
        else: self.final_solution_details = None
        if self.status == self.STATUS_OPTIMAL:
            try: self.final_sensitivity_data, self.final_interpretation_data = self._analyze_final_tableau()
            except Exception as e: print(f"ERREUR analyse: {e}"); self.final_sensitivity_data, self.final_interpretation_data = None, None
        else: self.final_sensitivity_data, self.final_interpretation_data = None, None
        if self.tableau is not None and self.status not in [self.STATUS_INFEASIBLE, self.STATUS_ERROR, self.STATUS_UNBOUNDED]:
            last_iter_title = self.iteration_results[-1]['title'] if self.iteration_results else ""
            status_already_logged = self.status in last_iter_title or self.STATUS_MAX_ITERATIONS in last_iter_title
            if not status_already_logged:
                phase_str = f"Phase {self.phase}" if self.needs_phase1 else ""
                title = f"{phase_str} - Tableau Final ({self.status})"
                try:
                    final_tab_str = self._format_tableau_to_string(self.tableau, phase=self.phase)
                    self.iteration_results.append({"title": title, "tableau_str": final_tab_str, "message": "Dernier tableau."})
                except Exception as e: print(f"ERREUR formatage tableau final: {e}")