from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
import random
import os
import gc

app = Flask(__name__)

model = keras.models.load_model('fix_model.h5')
user_id = 0
dot_product = np.genfromtxt('data/dot_prod.csv').astype(int)
preprocess_content_data = pd.read_csv('data/preprocess_content_based_data.csv', sep=',')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/user_input', methods=['POST'])
def user_input():
    global user_id
    new_user_id = int(request.form['user_id'])
    user_id = new_user_id
    data = preprocess_data()
    if new_user_id in data['user_id'].values:
        return render_template('user_input.html', id=new_user_id)
    else:
        return render_template('new_user_input.html', id=new_user_id)


@app.route('/recommend', methods=['POST'])
def recommend():
    liked_food_id = (request.form['liked_food_id'].split(','))
    recommended_items_id, recommended_items_name = get_recommendations(user_id, liked_food_id)
    food_id = []
    for id in recommended_items_id:
        parts = id.split('-')
        food_id.append(int(parts[1]))

    output_food_img_like = []
    for id in food_id:
        image_path = get_image_path(int(id))
        output_food_img_like.append(image_path)

    recommendations_json = {
        "user_id": user_id,
        "like_based_recommendation": recommended_items_name,
        "image_like_path": output_food_img_like,
    }

    del liked_food_id, recommended_items_id, recommended_items_name, food_id, output_food_img_like
    gc.collect()

    return jsonify(recommendations_json)



@app.route('/new_user_recommend', methods=['POST'])
def new_user_recommend():
    food_ids = [int(id) for id in request.form['food_id'].split(',')]
    similar_food_with_id = find_similar_food(food_ids)
    recommended_food_ids, recommended_food_names = zip(*similar_food_with_id)
    flattened_similar_food = list(zip(recommended_food_ids, recommended_food_names))
    random.shuffle(flattened_similar_food)
    simplified = flattened_similar_food[:10]
    output_food_id = []
    output_food_name = []
    output_food_img = []
    for id, name in simplified:
        image_path = get_image_path(int(id))
        output_food_id.append(id)
        output_food_name.append(name)
        output_food_img.append(image_path)


    liked_food_id = (request.form['liked_food_id'].split(','))
    recommended_items_id, recommended_items_name = get_recommendations(user_id, liked_food_id)
    food_id = []
    resto_id = []
    for id in recommended_items_id:
        parts = id.split('-')
        food_id.append(int(parts[1]))
        resto_id.append(int(parts[0]))

    output_food_img_like = []
    for id in food_id:
        image_path = get_image_path(int(id))
        output_food_img_like.append(image_path)

    recommendations_json = {
        "user_id": user_id,
        "image_content_path": output_food_img,
        "image_like_path" : output_food_img_like,
        "label_based_recommendation": output_food_name,
        "like_based_recommendation": recommended_items_name,
    }
    del food_ids, similar_food_with_id, recommended_food_ids, recommended_food_names, flattened_similar_food
    del simplified, output_food_id, output_food_name, output_food_img, liked_food_id, recommended_items_id, recommended_items_name, food_id, resto_id, output_food_img_like

    gc.collect()
    return jsonify(recommendations_json)

def get_image_path(food_id):
    return preprocess_content_data.loc[preprocess_content_data['food_id'] == food_id].reset_index(drop=True)['image'].iloc[0]


def find_similar_food(food_ids):
    recommend = []

    for food_id in food_ids:
        food_idx = np.where(preprocess_content_data['food_id'] == food_id)[0][0]
        similar_idxs = np.where(dot_product[food_idx] == np.max(dot_product[food_idx]))[0]
        similar_food = list(zip(preprocess_content_data.iloc[similar_idxs, ]['food_id'], preprocess_content_data.iloc[similar_idxs, ]['food_name']))
        recommend.extend(similar_food)

    return recommend


def preprocess_data():
    df = pd.read_csv('data/data_1d.csv', sep=';', header=[1])
    df.drop(df.columns[0:2], axis=1, inplace=True)
    df.drop(df.columns[3], axis=1, inplace=True)

    df.rename(
        columns={
            'ID': 'food_id',
            'MAKANN': 'food_name',
            'TEMPAT': 'resto_name'
        }, inplace=True
    )

    like_df = pd.read_csv('data/like_data.csv', sep=';')

    like_with_food = like_df.merge(df, on='food_id')

    like_with_food.drop(like_with_food.columns[3:], axis=1, inplace=True)

    food_id_data = df['food_id']
    food_id_df = pd.DataFrame(food_id_data)

    like_data = like_with_food
    like_data_df = pd.DataFrame(like_data)

    all_combinations_df = pd.DataFrame(
        [(user_id, food_id) for user_id in like_data_df['user_id'].unique() for food_id in food_id_df['food_id']],
        columns=['user_id', 'food_id'])

    merged_df = pd.merge(all_combinations_df, like_data_df, on=['user_id', 'food_id'], how='left')
    merged_df['like'] = merged_df['like'].fillna(0).astype(int)

    food_like = merged_df.merge(df, on='food_id')

    return food_like

def get_recommendations(new_user_id, liked_food_id):

    data = preprocess_data()

    user_encoder = LabelEncoder()
    food_encoder = LabelEncoder()

    data['user_encoded'] = user_encoder.fit_transform(data['user_id'])
    data['food_encoded'] = food_encoder.fit_transform(data['food_id'])

    user_encoder.fit(pd.concat([data['user_id'], pd.Series([new_user_id])]))
    user_encoder.transform([new_user_id])
    np.array([food_encoder.transform(liked_food_id)])

    new_user_model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=len(user_encoder.classes_) + 1, output_dim=50,
                                  name='user_embedding'),
        tf.keras.layers.Flatten()
    ])

    new_user_model.compile(optimizer='adam', loss='binary_crossentropy')
    new_user_embedding_weights = new_user_model.get_layer('user_embedding').get_weights()[0]
    user_embedding = model.get_layer('user_embedding').get_weights()[0]

    user_embedding /= np.linalg.norm(user_embedding, axis=1, keepdims=True)
    new_user_embedding_weights /= np.linalg.norm(new_user_embedding_weights, axis=1, keepdims=True)

    similarity_scores = cosine_similarity(
        user_embedding,
        new_user_embedding_weights
    )[0]

    top_similar_users_indices = np.argsort(similarity_scores)[::-1]

    liked_items_of_similar_users = data[
        (data['user_id'].isin(top_similar_users_indices[:6])) & (data['like'] == 1)].drop_duplicates()

    randomized_rows = liked_items_of_similar_users.sample(frac=1)

    recommended_randomized_food = randomized_rows[['food_id', 'food_name']].values.tolist()

    food_id_list = [food_id for food_id, _ in recommended_randomized_food[:6]]
    food_name_list = [food_name for _, food_name in recommended_randomized_food[:6]]

    gc.collect()

    return food_id_list, food_name_list

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)
