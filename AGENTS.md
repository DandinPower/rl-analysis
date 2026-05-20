# Python Virtual Environment

1. This repo use `uv venv` to maintain python virtual environment (not uv init or uv add, etc.).
2. For any python related commands, please activate the virtual environment first: `source .venv/bin/activate`. 
3. This repo already installed all the required packages in the virtual environment. 
4. Use `gitingest` under `rl/analysis` to generate a `digest.txt` file that contains all the necessary sourcecodes that can be used for report listing.

# LaTeX Writing Conventions

1. Avoid using enumerate or itemize environments in LaTeX documents. Instead, use simple paragraphs or sections to present information.
2. Avoid using fancy words or complex sentence structures in LaTeX documents. Aim for clarity and simplicity in writing.
3. Use section and subsection headings to organize the content of the LaTeX document. No more than 3 levels of headings (section, subsection, subsubsection) should be used.
4. can run `make` under the `report/` directory to compile the LaTeX document. The main file is `main.tex`, and the output PDF will be generated as `build/main.pdf`.