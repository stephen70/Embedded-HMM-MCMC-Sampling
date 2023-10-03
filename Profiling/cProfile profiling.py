import os

project_directory = ""  # enter path to the project directory here
target_file_path = project_directory + "Samplers/PGBSSampler.py"

os.system(f"python -m cProfile -o profiling.prof {target_file_path}")
os.system(f"snakeviz profiling.prof")