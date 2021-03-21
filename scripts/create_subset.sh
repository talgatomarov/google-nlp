#!/bin/sh

# run from scripts folder. I did not set up root folder properly
head -5000 ../data/articles.csv > ../data/articles_subset.csv
echo "Done."