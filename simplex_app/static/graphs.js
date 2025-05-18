document.addEventListener('DOMContentLoaded', function () {
    const numNodesInput = document.getElementById('num_nodes');
    const nodeNamesContainer = document.getElementById('nodeNamesContainer');
    const matrixContainer = document.getElementById('adjacencyMatrixContainer');
    const graphVisualizationContainer = document.getElementById('graphVisualization'); // Pour la visualisation

    // Valeurs initiales/soumises (passées par Flask via des variables JS globales dans le template)
    // Ces variables `submittedGraphData` et `graphDataForVisJs` doivent être définies dans le HTML
    // AVANT que ce script ne soit chargé, ou au moins avant que DOMContentLoaded ne se déclenche.
    const initialNumNodes = (typeof submittedGraphData !== 'undefined' && submittedGraphData.num_nodes) 
                            ? parseInt(submittedGraphData.num_nodes) 
                            : (numNodesInput ? parseInt(numNodesInput.value) : 3); // Valeur du champ ou défaut
    
    const initialNodeNames = (typeof submittedGraphData !== 'undefined' && submittedGraphData.node_names) 
                            ? submittedGraphData.node_names 
                            : {};
    
    const initialAdjMatrix = (typeof submittedGraphData !== 'undefined' && submittedGraphData.adj_matrix) 
                            ? submittedGraphData.adj_matrix 
                            : {};

    function generateGraphInputs() {
        if (!numNodesInput || !nodeNamesContainer || !matrixContainer) {
            console.error("Éléments DOM requis non trouvés pour la génération des inputs du graphe.");
            return;
        }

        const numNodes = parseInt(numNodesInput.value);
        nodeNamesContainer.innerHTML = ''; // Clear previous node name inputs
        matrixContainer.innerHTML = '';    // Clear previous matrix

        if (isNaN(numNodes) || numNodes <= 0) {
            nodeNamesContainer.innerHTML = '<p class="note"><i>Veuillez entrer un nombre de nœuds valide (entier positif).</i></p>';
            return;
        }
        if (numNodes > 12) { // Limite pour l'affichage et la performance
            nodeNamesContainer.innerHTML = '<p class="warning">Maximum 12 nœuds pour un affichage raisonnable de la matrice et une bonne performance de la visualisation.</p>';
            return;
        }

        // --- Générer les inputs pour les noms des nœuds ---
        let nodeNamesHtml = '<label style="margin-bottom:8px; display:block;">Noms des Nœuds (uniques, ex: A, B, C ou 0, 1, 2):</label><div>';
        for (let i = 0; i < numNodes; i++) {
            // Utiliser les valeurs initiales si disponibles, sinon générer A, B, C...
            const prevName = initialNodeNames[i] || String.fromCharCode(65 + i);
            nodeNamesHtml += `<input type="text" name="node_name_${i}" class="node-name-input" value="${prevName}" placeholder="Nœud ${i + 1}" required pattern="[a-zA-Z0-9_.-]+" title="Caractères autorisés : lettres, chiffres, underscore (_), point (.), tiret (-)"> `;
        }
        nodeNamesHtml += '</div>';
        nodeNamesContainer.innerHTML = nodeNamesHtml;

        // --- Générer la matrice d'adjacence ---
        let tableHtml = '<table class="matrix-table"><thead><tr><th> </th>'; // Cellule vide pour le coin
        const currentGivenNodeNames = [];
        nodeNamesContainer.querySelectorAll('input.node-name-input').forEach(input => {
            currentGivenNodeNames.push(input.value.trim() || `N${currentGivenNodeNames.length + 1}`);
        });

        currentGivenNodeNames.forEach(name => {
            tableHtml += `<th>${escapeHtml(name)}</th>`; // Échapper pour sécurité
        });
        tableHtml += '</tr></thead><tbody>';

        for (let i = 0; i < numNodes; i++) {
            tableHtml += `<tr><th>${escapeHtml(currentGivenNodeNames[i])}</th>`; // En-tête de ligne
            for (let j = 0; j < numNodes; j++) {
                const cellId = `${i}-${j}`;
                // Utiliser les valeurs initiales de la matrice si disponibles
                const prevValue = initialAdjMatrix[cellId] || (i === j ? '0' : ''); // 0 sur la diagonale par défaut
                const isDiagonal = (i === j);
                const placeholderText = `Poids ${escapeHtml(currentGivenNodeNames[i])} → ${escapeHtml(currentGivenNodeNames[j])}`;
                tableHtml += `<td><input type="number" step="any" name="adj_matrix_${i}_${j}" class="${isDiagonal ? 'diagonal' : ''}" value="${prevValue}" title="${placeholderText}"></td>`;
            }
            tableHtml += '</tr>';
        }
        tableHtml += '</tbody></table>';
        matrixContainer.innerHTML = tableHtml;

        // Ajouter des écouteurs pour mettre à jour les en-têtes de la matrice
        nodeNamesContainer.querySelectorAll('input.node-name-input').forEach(input => {
            input.addEventListener('input', updateMatrixHeaders);
            input.addEventListener('change', updateMatrixHeaders); // Au cas où 'input' ne se déclenche pas toujours
        });
    }
    
    function updateMatrixHeaders() {
        if (!numNodesInput || !nodeNamesContainer || !matrixContainer) return;
        const numNodes = parseInt(numNodesInput.value);
        if (isNaN(numNodes) || numNodes <= 0 || numNodes > 12) return;

        const nodeNameInputs = nodeNamesContainer.querySelectorAll('input.node-name-input');
        const matrixTable = matrixContainer.querySelector('.matrix-table');
        if (!matrixTable || nodeNameInputs.length !== numNodes) return;

        const headerCells = matrixTable.querySelectorAll('thead th'); // Le premier est vide (pour le coin)
        const rowHeaderCells = matrixTable.querySelectorAll('tbody tr th:first-child'); // En-têtes de lignes

        for (let i = 0; i < numNodes; i++) {
            const newName = escapeHtml(nodeNameInputs[i].value.trim() || `Nœud ${i + 1}`);
            if (headerCells[i + 1]) headerCells[i + 1].textContent = newName; // i+1 pour sauter la première cellule d'en-tête de colonne (vide)
            if (rowHeaderCells[i]) rowHeaderCells[i].textContent = newName;   // Mettre à jour l'en-tête de la ligne i

            // Mettre à jour les tooltips des inputs de la matrice
            for (let j = 0; j < numNodes; j++) {
                const otherNodeName = escapeHtml(nodeNameInputs[j].value.trim() || `Nœud ${j + 1}`);
                const matrixInputRow = matrixTable.querySelector(`input[name="adj_matrix_${i}_${j}"]`);
                if (matrixInputRow) matrixInputRow.title = `Poids de ${newName} vers ${otherNodeName}`;
                
                // Si la matrice est symétrique pour les tooltips (optionnel)
                const matrixInputCol = matrixTable.querySelector(`input[name="adj_matrix_${j}_${i}"]`);
                if (matrixInputCol) matrixInputCol.title = `Poids de ${otherNodeName} vers ${newName}`;
            }
        }
    }

    function escapeHtml(unsafe) {
        if (typeof unsafe !== 'string') return unsafe; // Ne pas échapper si ce n'est pas une chaîne
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }


    // --- Écouteurs d'événements pour le champ num_nodes ---
    if (numNodesInput) {
        numNodesInput.addEventListener('change', generateGraphInputs);
        numNodesInput.addEventListener('keyup', function(e) {
            // Générer si Entrée ou si la valeur est valide et a changé (pour éviter trop d'appels sur chaque frappe)
            const currentValue = parseInt(e.target.value);
            if (e.key === "Enter" || (currentValue > 0 && currentValue <= 12) ) {
                 if (currentValue !== parseInt(submittedGraphData?.num_nodes || 0) ) { // Évite régénération si valeur pas changée
                    generateGraphInputs();
                }
            }
        });
        // Définir la valeur initiale du champ num_nodes si elle vient de submittedGraphData
        if (typeof submittedGraphData !== 'undefined' && submittedGraphData.num_nodes) {
            numNodesInput.value = submittedGraphData.num_nodes;
        }
    }
    
    // Génération initiale au chargement de la page
    generateGraphInputs();
    // Mettre à jour les en-têtes une fois après la génération initiale si les noms étaient soumis
    if (typeof submittedGraphData !== 'undefined' && Object.keys(submittedGraphData.node_names || {}).length > 0) {
        updateMatrixHeaders();
    }


    // --- Visualisation du Graphe avec Vis.js ---
    // La variable `graphDataForVisJs` doit être définie dans le template HTML par Flask
    if (typeof graphDataForVisJs !== 'undefined' && graphDataForVisJs && graphVisualizationContainer) {
        try {
            const nodesArray = graphDataForVisJs.nodes || [];
            const edgesArray = graphDataForVisJs.edges || [];
            const isDirectedGraph = graphDataForVisJs.is_directed || false;
            const dijkstraPathEdgesArray = graphDataForVisJs.dijkstra_path_edges || [];

            console.log("Data for Vis.js:", graphDataForVisJs);

            // Créer les nœuds pour Vis.js
            const visNodes = new vis.DataSet(nodesArray.map(nodeId => ({ 
                id: String(nodeId), // Vis.js préfère les IDs en chaîne
                label: String(nodeId) 
            })));

            // Créer les arêtes pour Vis.js
            const visEdges = new vis.DataSet(edgesArray.map(edge => {
                let edgeObject = {
                    from: String(edge.from),
                    to: String(edge.to),
                    label: edge.weight !== undefined ? String(edge.weight) : '', // Afficher le poids/capacité
                    arrows: isDirectedGraph ? 'to' : undefined,
                    color: { color: '#848484', highlight: '#ff0000', hover: '#2B7CE9' },
                    font: {align: 'middle', strokeWidth: 0, color: '#444', size:12},
                    smooth: { type: 'continuous' } // Pour des arêtes plus directes si peu de nœuds
                };
                
                // Surligner les arêtes du chemin de Dijkstra
                const isDijkstraEdge = dijkstraPathEdgesArray.some(pEdge => 
                    (String(pEdge.from) === String(edge.from) && String(pEdge.to) === String(edge.to)) || 
                    (!isDirectedGraph && String(pEdge.from) === String(edge.to) && String(pEdge.to) === String(edge.from)) 
                );

                if (isDijkstraEdge) {
                    edgeObject.color = { color: '#e63946', highlight: '#e63946', hover: '#e63946' };
                    edgeObject.width = 3;
                    edgeObject.font = {align: 'middle', strokeWidth: 0, color: '#e63946', size:13, bold: {mod: 'bold'}};
                    edgeObject.chosen = { edge: function(values, id, selected, hovering) { values.width = 4; values.shadow = true; }};
                }
                return edgeObject;
            }));

            // Options pour le graphe
            const options = {
                layout: {
                    randomSeed: undefined, // Laisse Vis.js choisir ou spécifiez une graine
                    improvedLayout: true,
                    // hierarchical: { // Décommentez pour tester
                    //     enabled: false,
                    //     sortMethod: 'hubsize' // ou 'directed'
                    // }
                },
                edges: {
                    smooth: {
                         enabled: true,
                         type: "dynamic", // Bon compromis
                         roundness: 0.5
                    },
                    font: {
                        size: 12,
                        align: 'middle' // Peut être 'top', 'bottom'
                    },
                    color: {
                        inherit: false // Important pour que les couleurs personnalisées fonctionnent
                    },
                    widthConstraint: { maximum: 90 } // Limite la largeur du label de l'arête
                },
                nodes: {
                    shape: 'ellipse', // 'dot', 'box', 'circle', 'ellipse', 'database', 'text', 'diamond', 'star', 'triangle', 'triangleDown', 'hexagon', 'square'
                    size: 22,
                    font: {
                        size: 15,
                        color: '#ffffff', // Texte en blanc sur le nœud
                        face: 'Arial, Helvetica, sans-serif'
                    },
                    borderWidth: 2,
                    borderWidthSelected: 3,
                    color: {
                        border: '#0d6efd',      // Bleu
                        background: '#4dabf7',  // Bleu clair
                        highlight: { border: '#0a58ca', background: '#74c0fc' },
                        hover: { border: '#0a58ca', background: '#a5d8ff' }
                    },
                    shadow: {enabled: true, size:5, x:3, y:3}
                },
                physics: {
                    enabled: true, // Laisser activé pour une disposition initiale
                    solver: 'forceAtlas2Based', // 'barnesHut', 'repulsion', 'hierarchicalRepulsion', 'forceAtlas2Based'
                    forceAtlas2Based: {
                        gravitationalConstant: -40, // Plus petit = moins de répulsion forte
                        centralGravity: 0.005,       // Attire vers le centre
                        springLength: 100,         // Longueur désirée des arêtes
                        springConstant: 0.18,      // Rigidité des ressorts
                        damping: 0.4,              // Amortissement du mouvement
                        avoidOverlap: 0.5          // Pour éviter le chevauchement des nœuds (0 à 1)
                    },
                    stabilization: {
                        enabled: true,
                        iterations: 200, // Augmenter pour des graphes plus complexes si besoin
                        updateInterval: 25,
                        onlyDynamicEdges: false,
                        fit: true
                    },
                    adaptiveTimestep: true,
                    minVelocity: 0.75
                },
                interaction: {
                    hover: true,
                    tooltipDelay: 200,
                    navigationButtons: true, // Ajoute boutons de zoom/reset
                    keyboard: true, // Navigation clavier
                    dragNodes: true, // Permet de déplacer les nœuds
                    zoomView: true,
                    dragView: true
                },
                manipulation: false // Désactiver les outils d'édition du graphe
            };

            const data = { nodes: visNodes, edges: visEdges };
            const network = new vis.Network(graphVisualizationContainer, data, options);

            // Optionnel: désactiver la physique après un certain temps ou stabilisation
            network.on("stabilizationIterationsDone", function () {
                console.log("Graph stabilization finished.");
                network.setOptions( { physics: false } );
            });
             // Ou après un délai fixe
            // setTimeout(() => {
            //     network.setOptions( { physics: false } );
            //     console.log("Physics disabled after timeout.");
            // }, 5000); // Désactive après 5 secondes

        } catch (e) {
            console.error("Erreur lors de la création de la visualisation du graphe avec Vis.js:", e);
            if (graphVisualizationContainer) {
                graphVisualizationContainer.innerHTML = "<p style='color:red; padding:10px;'>Erreur lors du chargement de la visualisation du graphe. Vérifiez la console pour plus de détails.</p>";
            }
        }
    } else if (graphVisualizationContainer) {
        // S'il n'y a pas de graphDataForVisJs mais que le conteneur existe, on peut afficher un message.
        // Mais cela est déjà géré par le template Jinja (`{% if graph_data_for_js %}`)
    }
});