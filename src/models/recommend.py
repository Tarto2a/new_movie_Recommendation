import pandas as pd
from src.models.similarity import calculate_similarity
from src.data.preprocess import preprocess_text, expand_with_synonyms, detect_and_translate, correct_spelling

def limit_dataset_size(movie_data, sample_size=5000):
    if len(movie_data) > sample_size:
        movie_data = movie_data.head(sample_size)
    return movie_data

def recommend_movies(user_description, movie_data, top_n=5):
    import numpy as np

    if 'title' not in movie_data.columns or 'description' not in movie_data.columns:
        raise ValueError("The movie data must have 'title' and 'description' columns.")
    
    movie_data['processed_description'] = movie_data['processed_description'].fillna('No processed_description available')
    
    user_description = detect_and_translate(user_description)
    corrected_description = correct_spelling(user_description)
    processed_user_description = preprocess_text(corrected_description)
    expanded_input = expand_with_synonyms(processed_user_description)
    final_input = f"{processed_user_description} {expanded_input}"
    print(final_input)
    
    print("copying data ...")
    extended_data = movie_data.copy()
    extended_data.loc[len(extended_data)] = ["User Query", final_input, final_input]
    
    print("calculating similarity ...")
    similarity_matrix = calculate_similarity(extended_data['processed_description'].tolist())
    print("calculating similarity done ...")
    
    user_similarity_scores = similarity_matrix[-1][:-1]
    non_zero_indices = np.where(user_similarity_scores > 0)[0]
    non_zero_scores = user_similarity_scores[non_zero_indices]
    if len(non_zero_scores) == 0:
        return pd.DataFrame(columns=movie_data.columns)
    top_indices = non_zero_indices[np.argsort(non_zero_scores)[-top_n:][::-1]]
    return movie_data.iloc[top_indices]
