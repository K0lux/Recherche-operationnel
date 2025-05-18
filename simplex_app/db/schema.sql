
-- simplex_app/schema.sql
DROP TABLE IF EXISTS history;

CREATE TABLE history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
  problem_type TEXT NOT NULL DEFAULT 'simplex', -- 'simplex' ou 'graph'

  -- Champs pour Simplex (peuvent être NULL si problem_type = 'graph')
  objective_type TEXT,
  objective_coeffs TEXT,     -- JSON string
  constraints TEXT,          -- JSON string (list of dicts pour Simplex)
  
  -- Champs pour Graphes (peuvent être NULL si problem_type = 'simplex')
  graph_data_input_type TEXT, -- 'list' (pour textarea) ou 'matrix'
  graph_data TEXT,           -- Les données brutes (liste d'arêtes ou matrice sérialisée)
  graph_is_directed INTEGER, -- 0 pour false, 1 pour true
  graph_params TEXT,         -- JSON string pour {start_node, end_node, source_node, sink_node}
  graph_results TEXT,        -- JSON string des résultats des algos de graphe

  -- Champs communs pour les résultats
  status TEXT,               -- Statut final (Optimal, Infaisable, Non Borné, etc.)
  objective_value REAL,      -- Peut être NULL (pour Simplex ou si pas applicable pour Graphe)
  solution_vars TEXT,        -- JSON string (pour Simplex, peut être NULL)
  
  warning TEXT
);