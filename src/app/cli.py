import sys
from ..data.loader import load_processed_data
from ..models.recommend import recommend_movies, limit_dataset_size

def main():
    movie_data = load_processed_data()

    if movie_data is None:
        print("Error: Could not load the processed movie data.")
        sys.exit(1)

    user_description = input("Enter a description of the movie you want to watch: ").strip()

    if not user_description:
        print("Error: Description cannot be empty.")
        sys.exit(1)

    recommendations = recommend_movies(user_description, limit_dataset_size(movie_data))

    if recommendations.empty:
        print("No recommendations found based on your description.")
    else:
        print("\nTop movie recommendations based on your description:")
        for index, row in recommendations.iterrows():
            print(f"{row['title']} - {row['description']}")

if __name__ == "__main__":
    main()
