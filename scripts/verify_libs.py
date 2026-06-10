import librosa
import numpy as np
import scipy
import sklearn
import essentia
import demucs
import whisper
import torch
import httpx
import qdrant_client as qc
import typer
import rich
from rich.table import Table
from rich.markdown import Markdown
import jinja2

libs = {
    'librosa': (librosa.__name__, librosa.__version__),
    'numpy': (np.__name__, np.__version__),
    'scipy': (scipy.__name__, scipy.__version__),
    'sklearn': (sklearn.__name__, sklearn.__version__),
    'essentia': (essentia.__name__, essentia.__version__),
    'demucs': (demucs.__name__, demucs.__version__),
    'whisper': (whisper.__name__, whisper.__version__),
    'torch': (torch.__name__, torch.__version__),
    'httpx': (httpx.__name__, httpx.__version__),
    'qdrant_client': (qc.__name__, '-'),
    'typer': (typer.__name__, typer.__version__),
    'jinja2': (jinja2.__name__, jinja2.__version__),
}

table = Table(title='Library Versions')
table.add_column('Library', style='cyan', no_wrap=True)
table.add_column('Version', style='magenta')

for lib, (name, version) in libs.items():
    table.add_row(name, version)
rich.print(table)

import torch
device = '**Device**:\t\t'

if torch.cuda.is_available():
    device += 'cuda'

elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device += 'mps (Apple Silicon)'
else:
    device += 'cpu'
rich.print(Markdown(device + '\n'))
rich.print(Markdown('**GPU:**\t\t{}'.format(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')))
rich.print(Markdown('**Full Stack:**\t✅'))