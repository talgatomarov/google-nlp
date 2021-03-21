#!/bin/sh

# run from scripts folder. I did not set up root folder properly
touch ../data/articles.csv
cat ../data/articles1.csv > ../data/articles.csv
tail -n +2 ../data/articles2.csv >> ../data/articles.csv
tail -n +2 ../data/articles3.csv >> ../data/articles.csv
echo "Done."