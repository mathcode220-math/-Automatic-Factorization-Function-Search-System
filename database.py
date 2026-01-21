import sqlite3
import json

class FactorizationDB:
    def __init__(self, db_path="factorization_research_v2.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Functions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candidate_functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                code TEXT,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Results table (enhanced)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_id INTEGER,
                test_number TEXT,
                test_size INTEGER,
                score REAL,
                computation_time REAL,
                true_positives INTEGER,
                false_positives INTEGER,
                precision REAL,
                recall REAL,
                efficiency_score REAL,
                tested_k_values INTEGER,
                UNIQUE(function_id, test_number),
                FOREIGN KEY (function_id) REFERENCES candidate_functions(id)
            )
        """)

        # Test numbers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_numbers (
                number TEXT PRIMARY KEY,
                factors TEXT,
                difficulty_level INTEGER,
                is_prime BOOLEAN
            )
        """)

        self.conn.commit()

    def add_function(self, name, code, parameters=None):
        cursor = self.conn.cursor()
        params_json = json.dumps(parameters) if parameters else '{}'
        try:
            cursor.execute(
                "INSERT INTO candidate_functions (name, code, parameters) VALUES (?, ?, ?)",
                (name, code, params_json)
            )
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"Function {name} already exists")
            return None

    def record_result(self, function_id, test_number, score, time_ms, tp, fp, precision, recall, tested_k, efficiency_score=0.0):
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO test_results
            (function_id, test_number, test_size, score, computation_time,
             true_positives, false_positives, precision, recall, efficiency_score, tested_k_values)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (function_id, test_number, len(test_number), score, time_ms,
              tp, fp, precision, recall, efficiency_score, tested_k))
        self.conn.commit()

    def get_best_functions(self, min_score=0.5, limit=10):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cf.name, cf.code, AVG(tr.score) as avg_score,
                   AVG(tr.precision) as avg_precision,
                   AVG(tr.recall) as avg_recall,
                   COUNT(*) as test_count,
                   SUM(tr.true_positives) as total_tp,
                   SUM(tr.false_positives) as total_fp
            FROM candidate_functions cf
            JOIN test_results tr ON cf.id = tr.function_id
            WHERE tr.score >= ?
            GROUP BY cf.id
            HAVING avg_score > 0.3
            ORDER BY avg_score DESC, avg_precision DESC
            LIMIT ?
        """, (min_score, limit))
        return cursor.fetchall()

    def get_function_stats(self, function_name):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT
                AVG(score) as avg_score,
                MIN(score) as min_score,
                MAX(score) as max_score,
                AVG(precision) as avg_precision,
                AVG(recall) as avg_recall,
                COUNT(*) as tests_run
            FROM test_results tr
            JOIN candidate_functions cf ON tr.function_id = cf.id
            WHERE cf.name = ?
        """, (function_name,))
        return cursor.fetchone()

    def get_statistics(self):
        """Comprehensive database statistics"""
        cursor = self.conn.cursor()
        
        # Number of functions
        cursor.execute("SELECT COUNT(*) FROM candidate_functions")
        func_count = cursor.fetchone()[0]
        
        # Number of results
        cursor.execute("SELECT COUNT(*) FROM test_results")
        result_count = cursor.fetchone()[0]
        
        # Average results
        cursor.execute("SELECT AVG(score) FROM test_results")
        avg_score = cursor.fetchone()[0]
        
        # Best function
        cursor.execute("""
            SELECT cf.name, AVG(tr.score) as avg_score
            FROM candidate_functions cf
            JOIN test_results tr ON cf.id = tr.function_id
            GROUP BY cf.id
            ORDER BY avg_score DESC
            LIMIT 1
        """)
        best_func = cursor.fetchone()
        
        # Score distribution
        cursor.execute("""
            SELECT 
                CASE 
                    WHEN score >= 0.7 THEN '0.7+ (Excellent)'
                    WHEN score >= 0.5 THEN '0.5+ (Good)'
                    WHEN score >= 0.3 THEN '0.3+ (Acceptable)'
                    ELSE 'Below 0.3 (Poor)'
                END as category,
                COUNT(*) as count
            FROM test_results
            GROUP BY category
            ORDER BY category DESC
        """)
        score_distribution = cursor.fetchall()
        
        return {
            'total_functions': func_count,
            'total_results': result_count,
            'average_score': avg_score,
            'best_function': best_func,
            'score_distribution': score_distribution
        }

    def export_to_csv(self, filename="results_export.csv"):
        """Export results to CSV file"""
        import csv
        
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cf.name, cf.code, tr.test_number, tr.score, tr.computation_time, tr.true_positives, tr.false_positives, tr.precision, tr.recall
            FROM candidate_functions cf
            JOIN test_results tr ON cf.id = tr.function_id
        """)
        
        rows = cursor.fetchall()
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Function Name', 'Code', 'Test Number', 'Score', 'Computation Time', 'True Positives', 'False Positives', 'Precision', 'Recall'])
            writer.writerows(rows)
        
        return f"Exported {len(rows)} results to {filename}"