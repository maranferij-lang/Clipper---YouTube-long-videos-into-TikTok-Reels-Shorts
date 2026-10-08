"""patch_lightasd_cpu.py <path/to/Light-ASD/ASD.py>  — робить ASD.py придатним для CPU (без CUDA):
.cuda() → .to(DEV), torch.load(...) → map_location=DEV. Ідемпотентний. Бекап ASD.py.bak_gpu."""
import sys, re, shutil, os
p = sys.argv[1]; s = open(p, encoding='utf-8').read()
if 'DEV = ' in s: print('вже пропатчено'); sys.exit(0)
shutil.copy(p, p + '.bak_gpu')
s = s.replace('import torch', "import torch\nDEV = 'cuda' if torch.cuda.is_available() else 'cpu'", 1)
s = s.replace('.cuda()', '.to(DEV)')
s = re.sub(r'torch\.load\(([^)]*)\)', lambda m: f"torch.load({m.group(1)}, map_location=DEV)" if 'map_location' not in m.group(1) else m.group(0), s)
open(p, 'w', encoding='utf-8').write(s); print('пропатчено', p)
