# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 10:02:35 2025

@author:Shang Gao 
"""

import os
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import RtfFormatter

# Input and output filenames
input_file = "C:/Users/Laboratorio/MakeHologram/Con_Cor/FFT_SameNoiseBi T.py"
output_file = "C:/Users/Laboratorio/PhD_thesis/rft files/FFT_SameNoiseBi_T.rtf"

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
os.makedirs(output_dir, exist_ok=True)

# Read the Python file
with open(input_file, "r", encoding="utf-8") as f:
    code = f.read()

highlighted_code = highlight(
    code,
    PythonLexer(),
    RtfFormatter(style="default", fontface="Consolas", fontsize=16, fontweight="normal")
)
# Write the RTF output
with open(output_file, "w", encoding="utf-8") as f:
    f.write(highlighted_code)

print(f"✅ RTF file saved: {output_file}")
