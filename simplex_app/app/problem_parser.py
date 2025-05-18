def parse_problem_form(form):
    """
    Convertit les données du formulaire Flask en un dictionnaire problem_data
    utilisable par SimplexSolver. Retourne (problem_data, error_message).
    """
    try:
        objective_type = form.get('objective_type', 'max')
        num_vars = int(form.get('num_vars'))
        num_constraints = int(form.get('num_constraints'))

        if num_vars <= 0 or num_constraints <= 0:
            raise ValueError("Le nombre de variables et de contraintes doit être positif.")

        problem_data = {
            'objective': objective_type,
            'obj_coeffs': [],
            'constraints': [],
            'var_names': [f'x{i+1}' for i in range(num_vars)]
        }

        # Coefficients Objectif
        for i in range(num_vars):
            coeff_str = form.get(f'obj_coeff_{i}')
            if coeff_str is None or coeff_str.strip() == "":
                raise ValueError(f"Coefficient objectif manquant pour x{i+1}")
            problem_data['obj_coeffs'].append(float(coeff_str))

        # Contraintes
        for i in range(num_constraints):
            constraint_coeffs = []
            for j in range(num_vars):
                coeff_str = form.get(f'constraint_{i}_{j}')
                if coeff_str is None or coeff_str.strip() == "":
                    raise ValueError(f"Coefficient manquant pour Contrainte {i+1}, variable x{j+1}")
                constraint_coeffs.append(float(coeff_str))

            constraint_type = form.get(f'constraint_type_{i}', 'le')
            rhs_str = form.get(f'rhs_{i}')
            if rhs_str is None or rhs_str.strip() == "":
                raise ValueError(f"Membre droit (RHS) manquant pour Contrainte {i+1}")

            problem_data['constraints'].append({
                'coeffs': constraint_coeffs,
                'type': constraint_type,
                'rhs': float(rhs_str)
            })
        return problem_data, None # problem_data, error_message
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Erreur inattendue lors du parsing des données: {e}"
