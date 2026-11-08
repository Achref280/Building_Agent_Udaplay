# How to create a new repo in github:
On github click create new repo
in the project folder
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://..
git push -u origin main

# Environment:
install requirement
uv pip install -r requirements.txt
uv venv --python 3.13
source .venv/bin/activate
Deactivate venv and install other requirements file
deactivate is known if venv is already activated.
uv venv --clear .venv

# .gitignore
#python
#virtual environment
#secrets and local settings
#jupyter
#mac os

# chroma
Create the data base
create the embedding function
create the table (collection)
