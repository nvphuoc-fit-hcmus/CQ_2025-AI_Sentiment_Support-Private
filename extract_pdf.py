import subprocess, sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    import fitz
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pymupdf'], capture_output=True)
    import fitz

doc = fitz.open(r'd:\Phuoc\Nam_4\Do_An_Tot_Nghiep\Do_An_TN\docs\Đồ_Án_Tốt_Nghiệp_Safe_Alert (1).pdf')

output = []
output.append(f'Total pages: {len(doc)}')

toc = doc.get_toc()
if toc:
    output.append('\n=== TABLE OF CONTENTS ===')
    for item in toc:
        indent = '  ' * (item[0] - 1)
        output.append(f'{indent}{item[1]} (p.{item[2]})')

output.append('\n=== CONTENT ===')
for i in range(min(len(doc), 80)):
    text = doc[i].get_text()
    if text.strip():
        output.append(f'\n--- PAGE {i+1} ---')
        output.append(text[:3000])

with open(r'd:\Phuoc\Nam_4\Do_An_Tot_Nghiep\Do_An_TN\pdf_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(output))

print('Done! Output saved to pdf_output.txt')
