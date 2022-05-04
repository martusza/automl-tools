import timeit
import os

import pandas as pd
import autokeras as ak

train_path = os.path.join("data", "processed", "train.csv")
test_path = os.path.join("data", "processed", "test.csv")
target_col = "Survived"


def main(train_path,
         test_path,
         target_col,
         ):
    # Initialize the classifier.
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    clf = ak.StructuredDataClassifier(
        max_trials=1, directory="tmp_dir", overwrite=True
    )

    start_time = timeit.default_timer()
    # x is the path to the csv file. y is the column name of the column to predict.
    clf.fit(train_path, target_col)
    stop_time = timeit.default_timer()

    # Evaluate the accuracy of the found model.
    accuracy = clf.evaluate(test_path, target_col)[1]
    print("Accuracy: {accuracy}%".format(accuracy=round(accuracy * 100, 2)))
    print(
        "Total time: {time} seconds.".format(time=round(stop_time - start_time, 2))
    )


if __name__ == "__main__":
    main(train_path, test_path, target_col)