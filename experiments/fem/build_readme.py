from pathlib import Path
H=Path(__file__).resolve().parent
t=(H/"results_tables.md").read_text()
body=(H/"README_body.md").read_text() if (H/"README_body.md").exists() else ""
(H/"README.md").write_text((H/"README_head.md").read_text()+body+"\n\n## Appendix: all generated tables\n"+t+"\n")
