document.addEventListener('DOMContentLoaded', function() {
    const numVarsInput = document.getElementById('num_vars');
    const numConstraintsInput = document.getElementById('num_constraints');
    const dynamicInputsDiv = document.getElementById('dynamic-inputs');
    const form = document.getElementById('simplex-form'); // Récupérer le formulaire

    // Fonction pour obtenir une valeur de formulaire ou une valeur par défaut
    function getFormValue(fieldName, defaultValue = '') {
        if (form && form.elements[fieldName]) {
            return form.elements[fieldName].value;
        }
        // Si le formulaire a été soumis, Flask/Jinja pourrait avoir repeuplé les champs.
        // Mais pour les champs dynamiques, cela ne fonctionne pas directement.
        // Essayons de voir si le serveur a passé des données via des attributs data- (plus complexe)
        // ou si on a un objet request.form accessible via une variable JS globale (pas standard)
        // Pour l'instant, on se fie à ce qui est dans le DOM après rendu Jinja
        return defaultValue;
    }


    function generateInputs() {
        const numVars = parseInt(numVarsInput.value) || 0;
        const numConstraints = parseInt(numConstraintsInput.value) || 0;

        dynamicInputsDiv.innerHTML = ''; // Clear previous inputs

        if (numVars > 0 && numConstraints > 0) {
            // --- Objective Function ---
            const objectiveGroup = document.createElement('div');
            objectiveGroup.className = 'input-group';
            let objectiveHtml = `<label>Objectif Z =</label>`; // Changé le texte
            for (let i = 0; i < numVars; i++) {
                // Essayons de récupérer la valeur si le champ existait (après un post)
                // Mais comme les champs sont dynamiques, c'est difficile sans une logique côté serveur
                // pour passer les valeurs soumises au JS.
                // Pour l'instant, on les recrée vides si on ne peut pas les retrouver.
                // Pour le repeuplement après POST, les valeurs sont dans `request.form`
                // Jinja dans index.html a déjà mis la bonne valeur dans numVarsInput
                // Ici, on crée juste les champs. Le repeuplement des types se fera par le script dans index.html
                const val = document.querySelector(`input[name="obj_coeff_${i}"]`) ? document.querySelector(`input[name="obj_coeff_${i}"]`).value : '';
                objectiveHtml += `<input type="number" step="any" name="obj_coeff_${i}" placeholder="x${i+1}" value="${val}" required> `;
                 if (i < numVars - 1) {
                    objectiveHtml += ` + `;
                }
            }
            objectiveGroup.innerHTML = objectiveHtml;
            dynamicInputsDiv.appendChild(objectiveGroup);

            // --- Constraints ---
            const constraintsTitle = document.createElement('h3');
            constraintsTitle.textContent = 'Contraintes'; // Texte plus générique
            dynamicInputsDiv.appendChild(constraintsTitle);

            for (let i = 0; i < numConstraints; i++) {
                const constraintRow = document.createElement('div');
                constraintRow.className = 'constraint-row';
                constraintRow.innerHTML = `<label>C${i+1}:</label>`;

                const coeffDiv = document.createElement('div');
                coeffDiv.className = 'coeff-inputs';
                for (let j = 0; j < numVars; j++) {
                    const val = document.querySelector(`input[name="constraint_${i}_${j}"]`) ? document.querySelector(`input[name="constraint_${i}_${j}"]`).value : '';
                    coeffDiv.innerHTML += `<input type="number" step="any" name="constraint_${i}_${j}" title="C${i+1}, Coeff x${j+1}" placeholder="x${j+1}" value="${val}" required> `;
                     if (j < numVars - 1) {
                        coeffDiv.innerHTML += ` + `;
                    }
                }
                constraintRow.appendChild(coeffDiv);

                // --- SÉLECTEUR POUR LE TYPE DE CONTRAINTE ---
                let typeVal = 'le'; // Défaut
                const selectField = document.querySelector(`select[name="constraint_type_${i}"]`);
                if (selectField) {
                    typeVal = selectField.value;
                }

                // Sélecteur de type, créé vide, sera peuplé par le script dans index.html si nécessaire

                const selectHtml = `
                    <select name="constraint_type_${i}" title="C${i+1} Type">
                        <option value="le" ${typeVal === 'le' ? 'selected' : ''}><=</option>
                        <option value="ge" ${typeVal === 'ge' ? 'selected' : ''}>>=</option>
                        <option value="eq" ${typeVal === 'eq' ? 'selected' : ''}>=</option>
                    </select>`;
                constraintRow.innerHTML += selectHtml;

                const rhsVal = document.querySelector(`input[name="rhs_${i}"]`) ? document.querySelector(`input[name="rhs_${i}"]`).value : '';
                const rhsHtml = `<div class="rhs-input">
                                    <label for="rhs_${i}" class="sr-only">RHS C${i+1}</label>
                                    <input type="number" step="any" id="rhs_${i}" name="rhs_${i}" title="C${i+1} RHS" placeholder="RHS" value="" required>
                                 </div>`;
                constraintRow.innerHTML += rhsHtml;

                dynamicInputsDiv.appendChild(constraintRow);
            }
        } else {
            dynamicInputsDiv.innerHTML = '<p style="color: #666;"><i>Entrez le nombre de variables et de contraintes ci-dessus pour générer les champs de saisie.</i></p>';
        }
    }

    numVarsInput.addEventListener('change', generateInputs);
    numConstraintsInput.addEventListener('change', generateInputs);
    numVarsInput.addEventListener('keyup', generateInputs);
    numConstraintsInput.addEventListener('keyup', generateInputs);

    // Génération initiale
    // Si request.form existe dans le template, les valeurs de num_vars et num_constraints
    // seront déjà les bonnes. generateInputs() les utilisera.
    generateInputs();

    // Ajouter un petit script pour tenter de pré-remplir les types de contraintes si `request.form` est passé
    // Cela se fait après la génération initiale des champs.
    // Remarque: Ceci est une tentative de contournement. Idéalement, le serveur passerait les données
    // soumises d'une manière plus structurée au JavaScript si un repeuplement complet est nécessaire.


});

// Pour sr-only (si pas déjà dans votre CSS principal)
const style = document.createElement('style');
style.innerHTML = `.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }`;
document.head.appendChild(style);