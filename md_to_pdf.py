"""Markdown → HTML → PDF 全自动转换"""
import re, base64, os, subprocess, markdown
from pathlib import Path

MD_FILE = Path("report.md")
OUT_HTML = MD_FILE.with_suffix(".html")
OUT_PDF  = MD_FILE.with_suffix(".pdf")
BASE_DIR = Path(".")

def embed_images(text, base_dir):
    def replacer(m):
        alt, path = m.group(1), m.group(2)
        abs_path = base_dir / path
        if abs_path.exists():
            data = base64.b64encode(abs_path.read_bytes()).decode()
            ext = abs_path.suffix.lstrip(".")
            mime = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg"}.get(ext,"image/png")
            return f'![{alt}](data:{mime};base64,{data})'
        return m.group(0)
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replacer, text)

md_text = embed_images(MD_FILE.read_text(encoding="utf-8"), BASE_DIR)

extensions = ["tables","fenced_code","codehilite","toc","attr_list","md_in_html","pymdownx.arithmatex"]
extension_configs = {"pymdownx.arithmatex": {"generic": True}}
body_html = markdown.Markdown(extensions=extensions, extension_configs=extension_configs).convert(md_text)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>红外小目标检测 ACM 实验报告</title>
<script>
MathJax = {{
  tex: {{ inlineMath: [['$','$'],['\\\\(','\\\\)']], displayMath: [['$$','$$'],['\\\\[','\\\\]']], tags:'ams' }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre'] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: "SimSun","宋体",serif; font-size:14px; line-height:1.9; color:#1a1a1a; max-width:900px; margin:40px auto; padding:0 48px 80px; background:#fff; }}
  h1 {{ font-size:24px; text-align:center; border-bottom:2.5px solid #222; padding-bottom:10px; margin-bottom:8px; font-family:"SimHei","黑体",sans-serif; letter-spacing:2px; }}
  h1 + p {{ text-align:center; color:#555; font-size:13px; margin-bottom:4px; text-indent:0; }}
  h2 {{ font-size:17px; border-bottom:1.5px solid #bbb; padding-bottom:4px; margin-top:40px; margin-bottom:12px; font-family:"SimHei","黑体",sans-serif; }}
  h3 {{ font-size:15px; margin-top:22px; margin-bottom:8px; color:#222; font-family:"SimHei","黑体",sans-serif; }}
  p {{ margin:6px 0 8px; text-indent:2em; }}
  ul,ol {{ padding-left:2.2em; margin:6px 0 10px; }}
  li {{ margin:3px 0; }}
  table {{ border-collapse:collapse; width:100%; margin:14px 0 18px; font-size:13px; }}
  th,td {{ border:1px solid #c0c0c0; padding:7px 14px; text-align:left; }}
  th {{ background:#eef0f4; font-weight:bold; font-family:"SimHei","黑体",sans-serif; font-size:13px; }}
  tr:nth-child(even) td {{ background:#f8f8f8; }}
  td:first-child {{ font-weight:normal; }}
  pre {{ background:#f5f5f5; border:1px solid #ddd; border-radius:4px; padding:12px 16px; overflow-x:auto; font-size:12px; line-height:1.5; white-space:pre-wrap; word-break:break-all; margin:10px 0; }}
  code {{ font-family:"Consolas","Courier New",monospace; background:#f0f0f0; padding:1px 5px; border-radius:2px; font-size:12px; }}
  pre code {{ background:none; padding:0; }}
  img {{ max-width:100%; display:block; margin:20px auto; border:1px solid #ccc; border-radius:4px; box-shadow:0 2px 6px rgba(0,0,0,0.10); }}
  .arithmatex {{ overflow-x:auto; margin:10px 0; }}
  hr {{ border:none; border-top:1px solid #d0d0d0; margin:32px 0; }}
  strong {{ font-family:"SimHei","黑体",sans-serif; font-weight:bold; }}
  @media print {{
    body {{ margin:0; padding:18px 32px; max-width:none; font-size:12px; line-height:1.75; }}
    h1 {{ font-size:20px; }} h2 {{ font-size:15px; page-break-after:avoid; }} h3 {{ font-size:13px; page-break-after:avoid; }}
    pre {{ font-size:10px; page-break-inside:avoid; }}
    img {{ page-break-inside:avoid; max-height:380px; object-fit:contain; }}
    table {{ page-break-inside:avoid; font-size:11px; }}
    p {{ text-indent:2em; }}
  }}
</style>
</head>
<body>{body}</body>
</html>"""

OUT_HTML.write_text(HTML_TEMPLATE.format(body=body_html), encoding="utf-8")
print(f"HTML written: {OUT_HTML}")

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
html_uri = OUT_HTML.resolve().as_uri()
cmd = [CHROME, "--headless", "--disable-gpu",
       "--run-all-compositor-stages-before-draw",
       "--virtual-time-budget=12000",
       f"--print-to-pdf={OUT_PDF.resolve()}",
       html_uri]
result = subprocess.run(cmd, capture_output=True, timeout=45)
if OUT_PDF.exists():
    print(f"PDF OK: {OUT_PDF}  ({OUT_PDF.stat().st_size//1024} KB)")
else:
    print("PDF generation failed. stderr:", result.stderr.decode(errors='ignore')[:300])
    print("Fallback: open report.html in browser → Ctrl+P → Save as PDF")
