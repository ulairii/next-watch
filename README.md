# Next Watch

Movie suggestions from a few genres and optional star ratings: https://ulairii.github.io/next-watch/

The page runs entirely in the browser. Ratings and genres stay in the tab and are never sent anywhere.

## How it works

Five recommenders are trained on MovieLens ml-latest-small (100,835 ratings, 9,741 movies): popularity,
a TF-IDF genre model, item-item collaborative filtering (50 neighbors, similarity shrinkage 200),
PureSVD with 20 factors, and a hybrid. A visitor's ratings are folded into the trained models
without retraining. The site uses a genre blend with no ratings, a cold-start blend for 1 to 14
ratings, and the full hybrid from 15 ratings.

- `model/`: exported model files (movie table, genre vectors, item neighbors, SVD factors)
- `static/recommender.js`: scoring in the browser
- `static/main.js`, `static/style.css`, `index.html`: the interface

## Data

This site uses the MovieLens ml-latest-small dataset from GroupLens Research at the University of
Minnesota. The files in `model/` are derived from it and are distributed under the MovieLens license
in `model/MOVIELENS_README.txt`: non-commercial use only, and no endorsement by the University of
Minnesota or GroupLens is implied.

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1-19:19. https://doi.org/10.1145/2827872
