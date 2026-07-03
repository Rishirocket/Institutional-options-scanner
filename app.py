from flask import Flask, render_template, jsonify, request
from pathlib import Path
import json

from scanner import run_scan, STATE_FILE
from config import WHITELIST, DTE_BUCKETS, ALERT_SCORE_MIN

app = Flask(__name__)

@app.route('/')
def index():
    state = {}
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text())
        except Exception:
            state = {}
    return render_template('index.html', state=state, tickers=WHITELIST, dte=DTE_BUCKETS, score=ALERT_SCORE_MIN)

@app.route('/api/status')
def status():
    if STATE_FILE.exists():
        return jsonify(json.loads(STATE_FILE.read_text()))
    return jsonify({'status': 'No scan has run yet.'})

@app.route('/api/run', methods=['POST'])
def api_run():
    # For local/manual testing only. On free hosts this can time out.
    send_test = request.args.get('test', '0') == '1'
    return jsonify(run_scan(send_test=send_test))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
