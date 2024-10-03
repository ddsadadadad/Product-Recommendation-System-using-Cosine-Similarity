import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'product_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
    'rating': [5, 4, 0, 4, 3, 0, 2, 5, 3, 4]
}

df = pd.DataFrame(data)

product_data = {
    'product_id': [1, 2, 3, 4],
    'product_name': ['Продукт A', 'Продукт B', 'Продукт C', 'Продукт D']
}
products_df = pd.DataFrame(product_data)

user_product_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating').fillna(0)

print("Матрица Пользователь-Продукт:")
print(user_product_matrix)

similarity_matrix = cosine_similarity(user_product_matrix)

similarity_df = pd.DataFrame(similarity_matrix, index=user_product_matrix.index, columns=user_product_matrix.index)

print("\nМатрица Косинусного Сходства:")
print(similarity_df)

def get_recommendations(user_id, matrix, similarity):
    user_ratings = matrix.loc[user_id].values
    similar_users = similarity[user_id - 1]
    weighted_ratings = similar_users.dot(matrix)
    recommendations = weighted_ratings / np.array([np.abs(similar_users).sum()])
    recommendations_series = pd.Series(recommendations, index=matrix.columns)
    non_rated = recommendations_series[user_ratings == 0]
    return non_rated[non_rated > 0]

user_id = 1
recommendations = get_recommendations(user_id, user_product_matrix, similarity_matrix)
print(f"\nРекомендации для пользователя {user_id}:")

recommended_products = products_df.set_index('product_id').loc[recommendations.index]

print(recommended_products['product_name'])