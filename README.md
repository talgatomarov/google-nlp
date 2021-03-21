# Local Search Engine

Author: Talgat Omarov

## Set-up

Place dataset into data/ folder. You can either use 3 separate files (articles1.csv, articles2.csv, articles3.csv) or single merged file(articles.csv). If you are using 3 separate files, run scripts/merge_dataset.sh (from scripts folder)

You can either install dependencies globally

```
pip install -r requirements.txt
```

Or create virtual environemt

```
python3 -m venv venv
```

and then install dependencies.

Use

```
source ./venv/bin/activate
```

to activate the virtual environment

## Artifacts

To run local search engine you have to build artifacts. Run tfidf-svd.ipynb (from model/ folder) to build artifacts. I saved jupyter notebooks in .py format if you have issues with opening .ipynb files.

## Client

To run client appliation execute the following command from client/ folder

```
streamlit run app.py
```
