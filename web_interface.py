from flask import Flask, render_template, jsonify
import os
import sqlite3
from database import FactorizationDB

app = Flask(__name__)

@app.route('/')
def index():
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Factorization Function Search System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .stat-box {
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 10px;
            text-align: center;
            flex: 1;
            min-width: 200px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Factorization Function Search System</h1>
        
        <div class="stats">
            <div class="stat-box">
                <h3 id="total-functions">...</h3>
                <p>Total Functions</p>
            </div>
            <div class="stat-box">
                <h3 id="total-results">...</h3>
                <p>Total Results</p>
            </div>
            <div class="stat-box">
                <h3 id="avg-score">...</h3>
                <p>Average Score</p>
            </div>
            <div class="stat-box">
                <h3 id="best-function">...</h3>
                <p>Best Function</p>
            </div>
        </div>
        
        <button class="btn" onclick="loadFunctions()">Load Functions</button>
        <button class="btn" onclick="loadStats()">Refresh Stats</button>
        
        <table id="functions-table">
            <thead>
                <tr>
                    <th>Name</th>
                    <th>Avg F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Tests Count</th>
                    <th>TP</th>
                    <th>FP</th>
                </tr>
            </thead>
            <tbody id="functions-body">
            </tbody>
        </table>
    </div>

    <script>
        async function loadStats() {
            const response = await fetch('/api/stats');
            const stats = await response.json();
            
            document.getElementById('total-functions').textContent = stats.total_functions;
            document.getElementById('total-results').textContent = stats.total_results;
            document.getElementById('avg-score').textContent = stats.average_score ? stats.average_score.toFixed(3) : 'N/A';
            document.getElementById('best-function').textContent = stats.best_function ? stats.best_function[0] : 'N/A';
        }
        
        async function loadFunctions() {
            const response = await fetch('/api/functions');
            const functions = await response.json();
            
            const tbody = document.getElementById('functions-body');
            tbody.innerHTML = '';
            
            functions.forEach(func => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${func[0]}</td>
                    <td>${func[2].toFixed(3)}</td>
                    <td>${(func[3] * 100).toFixed(1)}%</td>
                    <td>${(func[4] * 100).toFixed(1)}%</td>
                    <td>${func[5]}</td>
                    <td>${func[6]}</td>
                    <td>${func[7]}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        // Load stats on page load
        window.onload = function() {
            loadStats();
        };
    </script>
</body>
</html>
    '''
    return html_content

@app.route('/api/functions')
def api_functions():
    db = FactorizationDB()
    functions = db.get_best_functions(min_score=0.0, limit=50)  # Get top 50 functions
    return jsonify(functions)

@app.route('/api/stats')
def api_stats():
    db = FactorizationDB()
    stats = db.get_statistics()
    return jsonify(stats)

if __name__ == '__main__':
    # Check if Flask is available
    try:
        app.run(debug=True, port=5000)
    except ImportError:
        print("Flask is not installed. Install it using: pip install flask")