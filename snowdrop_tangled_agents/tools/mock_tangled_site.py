"""
Local mock of tangled-game.com's Best-of-5 CHALLENGE, for testing
`play_tangled.py --challenge` without starting a live series.

It reproduces what the runner relies on, as read from the site's bundle
(2026-09-25): the gold `.challenge-button`, the banner "Game N of 5 · You are
Red/Blue" (odd games Red = P1), "Your turn", "Edges colored: k/15", SVG lines
at the /graph/5 vertex positions with the site's stroke colors, the
Grey/Green/Purple/Cancel dialog, the per-game "GAME N RESULT" overlay with
YOU WON! / YOU LOST / DRAW, the automatic start of the next game 3.2 s later,
and the final "YOU WIN THE CHALLENGE!" / "ALPHAQ UP WINS" / "SERIES DRAWN"
modal. Leaving mid-series posts /api/forfeit (the real page records the
remaining games as losses), which /mock/status reports.

The endpoints answer like the backend: /api/make_move (the AI's move is
chosen server-side), /api/interim_adjudicate, /api/adjudicate (lookup_table
winner and score) and /api/games/complete. The mock AI replays AlphaQ's
recorded replies from the game-end captures where it knows them (first free
edge, grey, otherwise); terminals score with the recorded table value where
known, the b4 model otherwise.

Usage:
    python -m snowdrop_tangled_agents.tools.mock_tangled_site [--port 8778]
"""

import argparse
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.tools import alphaq_captures as ac

PORT = 8778
POSITIONS = [[0.0, -100.0], [95.0, -31.0], [58.0, 80.0], [-59.0, 80.0], [-96.0, -31.0],
             [-210.0, -68.0], [0.0, -220.0], [209.0, -68.0], [129.0, 177.0], [-130.0, 177.0]]
SYM = {1: 'Z', 2: 'G', 3: 'P'}
LABEL = {'Z': 1, 'G': 2, 'P': 3}

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Tangled (mock)</title>
<style>body{font-family:sans-serif} .overlay{position:fixed;inset:0;background:rgba(0,0,0,.8);color:#fff;
display:flex;align-items:center;justify-content:center;flex-direction:column;font-size:28px}</style></head>
<body><div id="lobby"><h1>Play</h1><button class="challenge-button" onclick="startChallenge()">
<span>CHALLENGE</span><div>BEST OF 5 · PETERSEN GRAPH · VS ALPHAQ UP</div></button></div>
<div id="game" style="display:none">
 <div id="banner"></div><div id="status"></div><div id="count"></div><div id="score">Score: 0.000</div>
 <svg id="board" width="600" height="600"></svg><div id="dialog" style="display:none"></div>
</div><div id="ov"></div>
<script>
const EDGES=__EDGES__, POS=__POS__;
const STROKE={0:'#e5e7eb',1:'#9ca3af',2:'#10b981',3:'#a855f7'};
let n=0, human='player1', turn='player1', colors={}, gameId=null, results=[], busy=false, sel=null, inSeries=false, done=false;
async function api(path, body){const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});return r.json();}
function render(){
  const svg=document.getElementById('board'); svg.innerHTML='';
  EDGES.forEach((e,i)=>{const [a,b]=e; const l=document.createElementNS('http://www.w3.org/2000/svg','line');
    l.setAttribute('x1',300+POS[a][0]); l.setAttribute('y1',300+POS[a][1]); l.setAttribute('x2',300+POS[b][0]); l.setAttribute('y2',300+POS[b][1]);
    l.setAttribute('stroke',STROKE[colors[i]||0]); l.setAttribute('stroke-width','8'); l.addEventListener('click',()=>pick(i)); svg.appendChild(l);});
  const k=Object.keys(colors).length;
  document.getElementById('count').textContent='Edges colored: '+k+'/15';
  document.getElementById('status').textContent= k===15?'':(turn===human?'Your turn':"AlphaQ Up's turn");
  document.getElementById('banner').innerHTML='<b>CHALLENGE</b> Game '+n+' of 5 · You are '+(human==='player1'?'Red':'Blue');
}
function pick(i){ if(turn!==human||busy||colors[i]!==undefined) return; sel=i; const d=document.getElementById('dialog');
  d.innerHTML='<button onclick="choose(1)">Grey</button><button onclick="choose(2)">Green</button><button onclick="choose(3)">Purple</button><button onclick="cancelPick()">Cancel</button>';
  d.style.display='block'; }
function cancelPick(){sel=null; document.getElementById('dialog').style.display='none';}
async function choose(label){ const i=sel; cancelPick(); busy=true;
  await api('/api/make_move',{game_id:gameId,human_move:[0,i,label]}); colors[i]=label; await after(); busy=false; }
async function after(){ const s=await api('/api/interim_adjudicate',{game_id:gameId});
  document.getElementById('score').textContent='Score: '+s.interim_score.toFixed(3);
  if(Object.keys(colors).length===15){ render(); setTimeout(finish,500); return; }
  turn = turn==='player1'?'player2':'player1'; render();
  if(turn!==human) setTimeout(aiMove,500); }
async function aiMove(){ const m=await api('/api/make_move',{game_id:gameId,player_id:'ai'}); colors[m.edge_index]=m.color; await after(); }
async function finish(){
  const edges=EDGES.map((e,i)=>[e[0],e[1],colors[i]||0]);
  const adj=await api('/api/adjudicate',{game_id:gameId,game_state:{num_nodes:10,edges:edges,player1_node:5,player2_node:7}});
  const ours=human==='player1'?'red':'blue';
  const res= adj.winner==='draw'?'draw':(adj.winner===ours?'win':'loss');
  await api('/api/games/complete',{game_id:gameId,result:res,challenge_game:n});
  results.push({gameNum:n,result:res});
  const ov=document.getElementById('ov');
  ov.innerHTML='<div class="overlay"><div>GAME '+n+' RESULT</div><div>'+(res==='win'?'YOU WON!':res==='loss'?'YOU LOST':'DRAW')+'</div></div>';
  setTimeout(()=>{ ov.innerHTML=''; if(n<5) startGame(n+1); else final(); },3200);
}
function final(){ done=true; inSeries=false;
  const w=results.filter(r=>r.result==='win').length, l=results.filter(r=>r.result==='loss').length, d=results.filter(r=>r.result==='draw').length;
  const t= w>l?'\u{1F3C6} YOU WIN THE CHALLENGE!': l>w?'\u{1F614} ALPHAQ UP WINS':'\u{1F91D} SERIES DRAWN';
  document.getElementById('ov').innerHTML='<div class="overlay"><div>CHALLENGE</div><div>'+t+'</div><div>You '+w+' · '+l+' AlphaQ Up'+(d?' · '+d+' draw'+(d===1?'':'s'):'')+'</div></div>';
  api('/api/series_done',{wins:w,losses:l,draws:d,text:t}); }
async function startGame(k){ n=k; human= k%2===1?'player1':'player2'; turn='player1'; colors={};
  const g=await api('/api/initialize_game',{human:human,challenge_game:k}); gameId=g.game_id; render();
  if(turn!==human) setTimeout(aiMove,800); }
async function startChallenge(){ inSeries=true; document.getElementById('lobby').style.display='none';
  document.getElementById('game').style.display='block'; await startGame(1); }
window.addEventListener('beforeunload',()=>{ if(inSeries&&!done) navigator.sendBeacon('/api/forfeit',JSON.stringify({game:n})); });
</script></body></html>"""


class MockState:
    def __init__(self):
        games = ac.load_games()
        self.replies = {1: ac.reply_table(games, 2), 2: ac.reply_table(games, 1)}   # AI seat -> replies
        self.known = ac.known_terminals(games)
        self.games = {}
        self.completed = []
        self.forfeits = []
        self.series = None
        self.lock = threading.Lock()


STATE = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/mock/status"):
            return self._json({"completed": STATE.completed, "forfeits": STATE.forfeits, "series": STATE.series})
        page = PAGE.replace("__EDGES__", json.dumps([list(e) for e in tm.EDGES])).replace("__POS__", json.dumps(POSITIONS))
        data = page.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        path = self.path.split("?")[0]
        with STATE.lock:
            if path == "/api/initialize_game":
                gid = str(uuid.uuid4())
                STATE.games[gid] = {"board": ['-'] * 15, "human": body.get("human"), "n": body.get("challenge_game")}
                return self._json({"game_id": gid, "status": "created"})
            g = STATE.games.get(body.get("game_id"))
            if path == "/api/make_move":
                board = g["board"]
                if "human_move" in body:
                    _, e, label = body["human_move"]
                    if board[e] != '-':
                        return self._json({"detail": "edge already colored"}, 400)
                    board[e] = SYM[label]
                    return self._json({"move_type": 0, "edge_index": e, "color": label})
                ai_seat = 2 if g["human"] == "player1" else 1
                state = ''.join(board)
                mv = STATE.replies[ai_seat].get(state)
                if mv is None:
                    mv = (state.index('-'), 'Z')
                board[mv[0]] = mv[1]
                return self._json({"move_type": 0, "edge_index": mv[0], "color": LABEL[mv[1]]})
            if path == "/api/interim_adjudicate":
                return self._json({"interim_score": tm.score_state(''.join(g["board"]))})
            if path == "/api/adjudicate":
                t = ''.join(g["board"])
                score = STATE.known.get(t, tm.score_state(t))
                winner = "red" if score > tm.DRAW_EPSILON else "blue" if score < -tm.DRAW_EPSILON else "draw"
                return self._json({"game_state": dict(body.get("game_state", {}), player1_id="mock-user",
                                                      player2_id="AlphaQ Up"),
                                   "adjudicator": "lookup_table", "winner": winner, "score": score,
                                   "parameters": {"epsilon": tm.DRAW_EPSILON, "graph_number": 5}})
            if path == "/api/games/complete":
                STATE.completed.append({"game": g["n"], "result": body.get("result"), "terminal": ''.join(g["board"])})
                return self._json({"result": body.get("result"), "player_elo_before": 1000, "player_elo_after": 1000,
                                   "opponent_elo_before": 1200, "opponent_elo_after": 1200})
            if path == "/api/series_done":
                STATE.series = body
                return self._json({"ok": True})
            if path == "/api/forfeit":
                STATE.forfeits.append(body)
                return self._json({"ok": True})
        return self._json({"detail": "not found"}, 404)


def serve(port: int = PORT) -> ThreadingHTTPServer:
    global STATE
    STATE = MockState()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    server = serve(args.port)
    print(f"mock tangled site on http://127.0.0.1:{args.port}/play (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
