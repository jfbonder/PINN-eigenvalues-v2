from pathlib import Path
H=Path(__file__).resolve().parent
t=(H/"results_tables.md").read_text()
body=(H/"README_body.md").read_text() if (H/"README_body.md").exists() else ""
(H/"README.md").write_text((H/"README_head.md").read_text()+body+(H/"README_round2_head.md").read_text()+(H/"README_round2.md").read_text().split("\n",1)[1].replace("### 8.","### 8.",1)+"\n\n## Appendix: all generated tables\n"+t+"\n")
