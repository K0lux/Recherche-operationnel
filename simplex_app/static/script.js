document.addEventListener('DOMContentLoaded', function() {
    const numVarsInput = document.getElementById('num_vars');
    const numConstraintsInput = document.getElementById('num_constraints');
    const dynamicInputsDiv = document.getElementById('dynamic-inputs');

    // Function to generate input fields
    function generateInputs() {
        const numVars = parseInt(numVarsInput.value) || 0;
        const numConstraints = parseInt(numConstraintsInput.value) || 0;

        // Clear previous inputs
        dynamicInputsDiv.innerHTML = '';

        if (numVars > 0 && numConstraints > 0) {
            // --- Objective Function ---
            const objectiveGroup = document.createElement('div');
            objectiveGroup.className = 'input-group';
            let objectiveHtml = `<label>Objective (Z):</label>`;
            for (let i = 0; i < numVars; i++) {
                objectiveHtml += `<input type="number" step="any" name="obj_coeff_${i}" placeholder="x${i+1}" required> `;
            }
             // Placeholder for the operation (Max Z = ... or Min W = ...)
            objectiveGroup.innerHTML = objectiveHtml;
            dynamicInputsDiv.appendChild(objectiveGroup);


            // --- Constraints ---
            const constraintsTitle = document.createElement('h3');
            constraintsTitle.textContent = 'Constraints (Ax <= b)';
            dynamicInputsDiv.appendChild(constraintsTitle);

            for (let i = 0; i < numConstraints; i++) {
                const constraintRow = document.createElement('div');
                constraintRow.className = 'constraint-row';

                let constraintHtml = `<label>Constraint ${i+1}:</label><div class="coeff-inputs">`;
                for (let j = 0; j < numVars; j++) {
                    constraintHtml += `<input type="number" step="any" name="constraint_${i}_${j}" placeholder="x${j+1}" required> `;
                }
                 constraintHtml += `</div> <= <div class="rhs-input">`; // Assume <= for simplicity now
                 constraintHtml += `<input type="number" step="any" name="rhs_${i}" placeholder="RHS" required></div>`;

                constraintRow.innerHTML = constraintHtml;
                dynamicInputsDiv.appendChild(constraintRow);
            }
        }
    }

    // Add event listeners to update inputs when numbers change
    numVarsInput.addEventListener('change', generateInputs);
    numConstraintsInput.addEventListener('change', generateInputs);
    numVarsInput.addEventListener('keyup', generateInputs); // Update as user types
    numConstraintsInput.addEventListener('keyup', generateInputs);

    // Initial generation if values are pre-filled (e.g., after form submission error)
    generateInputs();
});