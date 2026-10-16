"""Embed output/app_data.json into app/edge_board_template.html.
  python3 build_page.py                      -> app/edge_board.html (Claude artifact copy, no live fetch)
  python3 build_page.py --live --out PATH    -> hosted copy that fetches live odds in the browser (GitHub Pages)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
live = "--live" in sys.argv
out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(HERE, "app", "edge_board.html")
b = json.load(open(os.path.join(HERE, "output", "app_data.json")))
for g in b["nfl"]["games"]:
    g["game_key"] = f"2026_06_{g['away']}_{g['home']}"
tpl = open(os.path.join(HERE, "app", "edge_board_template.html")).read()
html = tpl.replace("__DATA__", json.dumps(b, separators=(",", ":")).replace("</", "<\\/")).replace("__LIVE__", "true" if live else "false")
if live:  # standalone site: wrap in a full document (the artifact host adds this skeleton itself)
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>'
            '</head><body>' + html + '</body></html>')
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
open(out, "w").write(html)
print(out, len(html), "bytes; live" if live else "bytes; snapshot", "; built", b["generated_at"])
