import os, pathlib, sys
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath('d:/github/Drishya')]
import brisque
p = pathlib.Path(brisque.__file__)
print('brisque __init__ file:', p)
sub = p.parent / 'brisque.py'
print('module file:', sub)
print(sub.read_text()[:4000])
