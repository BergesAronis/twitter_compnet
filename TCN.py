import numpy as np
import pandas as pd
import hypothesis as H
import pickle
import TrainingSetConstructor

df = TrainingSetConstructor.get_dataset()
# df = pickle.load(open("temp_data.p", "rb"))
# # pickle.dump(df, open("temp_data.p", "wb"))
for column in df:
    df[column] = pd.to_numeric(df[column])
target = "DPZ"
train_data = df[["MCD", "YUM", "QSR", "DPZ", target + "_y"]].dropna()
print
alpha = 0.018299999999999945
iters = 1500

h1 = H.hypothesis([1], [0, 2], target + "_y", [0, 0, 0, 0])
h2 = H.hypothesis([0], [1, 2], target + "_y", [0, 0, 0, 0])
h3 = H.hypothesis([2], [0, 1], target + "_y", [0, 0, 0, 0])

h1.train(train_data, alpha, iters)
h2.train(train_data, alpha, iters)
h3.train(train_data, alpha, iters)

target = "MCD"

testing_set = df[["DPZ", "YUM", "QSR", "MCD", target + "_y"]].dropna()

print("Predicting outcomes...")
predicted_output = []
predictions = []

for index, row in testing_set.iterrows():
    x_array = np.array(row)[:-1]
    print(x_array)
    predict = [h1.predict(x_array), h2.predict(x_array), h3.predict(x_array)]
    predictions.append(predict)
    predicted_output.append(np.argmax(predict))

print(predicted_output)

errors = [0,0,0]
print("Calculating error")
sum_error = 0
expected_output = testing_set[target+"_y"].tolist()
for i in range(len(expected_output)):
    if predicted_output[i] != expected_output[i]:
        error = 1
        errors[int(expected_output[i])] += 1
    else:
        error = 0
    sum_error += error

print(errors)

avg_error = sum_error/len(predicted_output)

print("Average Error: " + str(round(100 * avg_error, 2)) + "%")
need_save = input("Save Models? (y/n):")

if need_save == "y":
    print("Genereating pickles...")
    pickle.dump(h1, open("h1.p", "wb"))
    pickle.dump(h2, open("h2.p", "wb"))
    pickle.dump(h3, open("h3.p", "wb"))
    print("Hypothesis saved")
