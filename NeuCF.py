# num_epochs = 
# batch_size = 
# mf_dim = 
# layers = 
# reg_mf = 
# reg_layers = 
# num_negatives = 
# learning_rate = 
# learner = 
# verbose = 
# mf_pretrain = 
# mlp_pretrain = 

# output = 
# dataset =



def ncf_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
  num_layers = len(layers)

  user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
  item_input = Input(shape=(1,), dtype='int32', name = 'item_input')


  # MLP embedding item
  item_embedding_mlp = keras.layers.Embedding(num_items + 1, n_latent_factors_item, name='item_embedding_MLP')(item_input)
  item_vec_mlp = keras.layers.Flatten(name='flatten_item_MLP')(item_embedding_mlp)
  item_vec_mlp = keras.layers.Dropout(0.2)(item_vec_mlp)


  # MF embedding item
  item_embedding_mf = keras.layers.Embedding(num_items + 1, n_latent_factors_mf, name='item_embedding_MF')(item_input)
  item_vec_mf = keras.layers.Flatten(name='flatten_item_MF')(item_embedding_mf)
  item_vec_mf = keras.layers.Dropout(0.2)(item_vec_mf)


  # MLP embedding user
  user_embedding_mlp = keras.layers.Embedding(num_users + 1, n_latent_factors_user, name='user_embedding_MLP')(user_input)
  user_vec_mlp = keras.layers.Flatten(name='flatten_user_MLP')(user_embedding_mlp)
  user_vec_mlp = keras.layers.Dropout(0.2)(user_vec_mlp)


  # MF embedding user
  user_embedding_mf = keras.layers.Embedding(num_users + 1, n_latent_factors_mf, name='user_embedding_MF')(user_input)
  user_vec_mf = keras.layers.Flatten(name='flatten_user_MF')(user_embedding_mf)
  user_vec_mf = keras.layers.Dropout(0.2)(user_vec_mf)

  # MF layer
  mf_vec = keras.layers.Multiply()([item_vec_mf, user_vec_mf])

  # MLP layer
  for idx in range(1, num_layer):
    layer = Dense(layers[idx], kernel_regularizer = l2(reg_layers[idx]),  bias_regularizer = l2(reg_layers[idx]),  activation='relu', name="layer%d" %idx)
    mlp_vec = layer(mlp_vec)
  

  predict_vec = keras.layers.Concatenate(axis=-1)([mf_vec, mlp_vec])
  prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_initializer ='lecun_uniform', name = "prediction")(predict_vec)

  model = Model([user_input, item_input], prediction)

  return model

def data():
  no_dup_u_i_r = pd.read_csv('/content/drive/My Drive/dataset/foody_rating.csv', usecols=['user_id', 'brand_id', 'avg_score']).drop_duplicates(keep='first').reset_index()
  no_dup_u_i = pd.read_csv('/content/drive/My Drive/dataset/foody_rating.csv', usecols=['user_id', 'brand_id']).drop_duplicates(keep='first').reset_index()
  dataset = no_dup_u_i_r

  # seperate test and train data later and set num_user to num user of train data
  # train and test should be 80 20, (1 user rate 5 restaurant = 4 train, 1 test?)

  num_users = len(dataset.user_id.unique())
  num_items = len(dataset.brand_id.unique())
  return user_id, brand_id


if __name__ == '__main__':

  model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(dataset, mf_dim, layers, time())

  # Loading data
  train, test = data()
  user_input = train.user_id
  item_input = train.brand_id

      
  # Build model
  model = ncf_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
  if learner.lower() == "adagrad": 
      model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
  elif learner.lower() == "rmsprop":
      model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
  elif learner.lower() == "adam":
      model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
  else:
      model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')

  # Training model
  for epoch in range(num_epochs):
      t1 = time()
      # Generate training instances
      user_input, item_input, labels = get_train_instances(train, num_negatives)
      
      # Training
      hist = model.fit([np.array(user_input), np.array(item_input)], #input
                        np.array(labels), # labels 
                        batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

  if args.out > 0:
      print("The best NeuMF model is saved to %s" %(model_out_file))











