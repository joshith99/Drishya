import sys, os
cwd = os.path.abspath('d:/github/Drishya')
sys.path = [p for p in sys.path if p and os.path.abspath(p) != cwd]
try:
    from brisque import NIQE
    print('NIQE available in brisque package')
except ImportError as e:
    print(f'NIQE not available: {e}')
