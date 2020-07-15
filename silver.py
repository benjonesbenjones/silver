from hyperopt import Trials, STATUS_OK, tpe
from keras.initializers import TruncatedNormal
from keras.utils import np_utils, to_categorical
from keras.layers import Dense, Activation, Dropout, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from hyperas import optim
from hyperas.distributions import choice, uniform
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv( 'lib/votes.csv', delimiter = ',')


y = data[['Clinton','Trump']]
data = data.drop( labels = ['state_abbreviation', 'area_name', 'Unnamed: 0', 'X', 'combined_fips', 'votes_dem_2016', 'votes_gop_2016', 'Clinton', 'Trump', 'diff_2016', 'per_point_diff_2016', 'state_abbr', 'county_name', 'fips', 'state_fips', 'total_votes_2012', 'votes_dem_2012', 'votes_gop_2012', 'county_fips', 'state_fips', 'Obama', 'Romney', 'diff_2012', 'diff_2016', 'per_point_diff_2012', 'per_point_diff_2016', 'Clinton_Obama', 'Trump_Romney', 'Trump_Prediction', 'Clinton_Prediction', 'Trump_Deviation', 'Clinton_Deviation', 'FIPS'], axis = 'columns')
x = data


x = x.values
y = y.values
print(np.shape(y))



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)

def data():

    data = pd.read_csv( 'lib/votes.csv', delimiter = ',')


    y = data[['Clinton','Trump']]
    data = data.drop( labels = ['state_abbreviation', 'area_name', 'Unnamed: 0', 'X', 'combined_fips', 'votes_dem_2016', 'votes_gop_2016', 'Clinton', 'Trump', 'diff_2016', 'per_point_diff_2016', 'state_abbr', 'county_name', 'fips', 'state_fips', 'total_votes_2012', 'votes_dem_2012', 'votes_gop_2012', 'county_fips', 'state_fips', 'Obama', 'Romney', 'diff_2012', 'diff_2016', 'per_point_diff_2012', 'per_point_diff_2016', 'Clinton_Obama', 'Trump_Romney', 'Trump_Prediction', 'Clinton_Prediction', 'Trump_Deviation', 'Clinton_Deviation', 'FIPS'], axis = 'columns')
    x = data


    x = x.values
    y = y.values
    print(np.shape(y))



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = 42)

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    init = TruncatedNormal(mean=0.0, stddev=0.05, seed=None)

    model = Sequential()

    model.add(Dense(52, input_dim=52))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))
    model.add(Dense({{choice([32, 64, 128, 256, 512, 1024])}}))
    model.add(BatchNormalization())
    model.add(Activation({{choice(['relu', 'sigmoid', 'softmax'])}}))

    model.add(Dense(2, activation='sigmoid'))


    model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.1, decay=1e-6),
              metrics=['mse'])


    model.fit(x_train, y_train, epochs=10, batch_size=1000, validation_split=0.1,
          class_weight={0:0.78, 1:0.22},
         )
    score = model.evaluate(x_test, y_test, verbose=0)
    mean_squared_error = score[1]
    accuracy = score[1]
    return {'loss': -mean_squared_error, 'status': STATUS_OK, 'model': model}


best_run, best_model = optim.minimize(model=create_model,
                          data=data,
                          algo=tpe.suggest,
                          max_evals=10,
                          trials=Trials())



print("Evaluation of best performing model:")
print(best_model.evaluate(x_test, y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)

best_model.save('lib/genetic-algorithm.h5')
