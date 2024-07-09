# CINEAI - Multimodal Siamese Network for Movie Genre Classification

## Overview
CINEAI implements a multimodal Siamese network for classifying movie genres using both movie poster images and plot descriptions. The model employs a combination of Convolutional Neural Networks (CNNs) for image processing and Long Short-Term Memory networks (LSTMs) for text processing to judge class similarity between pairs of movies.

## Dataset
The dataset includes movie poster images and corresponding plot text descriptions for four genres: Comedy, Horror, Romance, and Action. The dataset can be downloaded from Kaggle.

### Directory Structure
IMDB_four_genre_posters/
├── Comedy/
├── Horror/
├── Romance/
└── Action/


### CSV File
The CSV file `IMDB_four_genre_larger_plot_description.csv` contains the plot descriptions with columns `movie_id` and `description`.

## Dependencies
The following libraries are required to run the notebook:
- TensorFlow
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

## Model Architecture
The model consists of:

CNN block for processing images
LSTM block for processing text descriptions
Concatenation of features from both modalities
Dense layers for final classification

## Acknowledgements
This project uses data from Kaggle: IMDB Multimodal Vision and NLP Genre Classification.

## License
This project is licensed under the MIT License.

Install the dependencies using:
```bash
pip install tensorflow numpy opencv-python scikit-learn matplotlib
