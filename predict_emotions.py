# Dataset used (Audio_Song_Actors_01-24.zip):
# https://zenodo.org/record/1188976
# Resources used:
# https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#forceEdit=true&sandboxMode=true&scrollTo=-HJV4JF789aC
# https://towardsdatascience.com/extract-features-of-music-75a3f9bc265d
# https://www.kdnuggets.com/2020/02/audio-data-analysis-deep-learning-python-part-1.html

from sklearn.preprocessing import StandardScaler
import numpy as np
import keras
import os
import glob
import librosa
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # stop attempted use of GPU


def predict_emotions():
    # NOTE: "Surprised" is not represented in the the audio dataset so there are 7 emotions to train for
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust"]

    # lists used to hold the features and labels of all the parsed files
    all_feat = []
    all_lab = []
    all_files = []

    # Data pre-processing: Looks for all audio files of each actor within the current directory under folder "data"
    actor_paths = [x[0] for x in os.walk(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))]
    actor_paths.pop(0)
    for path in actor_paths:
        for filepath in glob.glob(os.path.join(path, "*.wav")):
            features = []  # holds all features for one audio file
            y, sr = librosa.load(filepath, sr=None)
            chroma_shft = librosa.feature.chroma_stft(y=y, sr=sr)  # captures harmonic/melodic shifts
            rms = librosa.feature.rms(y=y)[0]  # calculates the "intensity" of the sound
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)  # "center of mass" of the sound
            spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)  # frequency at which 85% of the energy is below it
            zcr = librosa.feature.zero_crossing_rate(y)  # number of times signal has crossed 0
            mfcc = librosa.feature.mfcc(y=y, sr=sr)  # 20 features which together describe the spectral envelope

            # takes the mean of each feature to get the average value across all frames
            features.extend([np.mean(chroma_shft), np.mean(rms), np.mean(spec_cent), np.mean(spec_roll), np.mean(zcr)])
            for f in mfcc:
                features.append(np.mean(f))

            # populates the feature and label lists
            filename = os.path.basename(filepath)
            all_feat.append(features)
            all_lab.append(int(filename.split("-")[2]))  # pulls emotion label from file name
            all_files.append(filename)

            print("Parsed audio file: ", filename)

    # standardize the feature values so weights and biases have a bigger impact
    scaler = StandardScaler()
    all_feat = scaler.fit_transform(np.array(all_feat, dtype=float))

    # split the files into training and testing lists
    perc_train = .80
    num_train = math.floor(len(all_files) * perc_train)

    train_feat = np.asarray(all_feat[:num_train])
    train_lab = np.asarray(all_lab[:num_train])

    test_feat = np.asarray(all_feat[num_train:])
    test_lab = np.asarray(all_lab[num_train:])

    # train = all_files[:num_train]
    # test = all_files[num_train:]
    # print("Training files: \n", train)
    # print("Testing files: \n", test)

    # neural network model creation
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(25,)),  # 25 input nodes to match the number of features
        keras.layers.Dense(15, activation="relu"),
        keras.layers.Dense(7, activation="softmax")  # 7 output nodes to match the number of emotions
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # neural network training
    model.fit(train_feat, train_lab, epochs=8)

    model.summary()

    # following commented code outputs the prediction vs actual emotions of the test set
    # predictions = model.predict(test_feat)
    # success = 0
    # for x in range(0, len(predictions)):
    #     predicted = emotions[np.argmax(predictions[x])]
    #     actual = emotions[int(test_lab[x])]
    #     print(x, " Predicted: ", predicted, " Actual: ", actual)
    #     if predicted == actual:
    #         success += 1
    # print("Prediction success: ", (success/len(predictions)))

    # print accuracy of test set
    acc = model.evaluate(test_feat, test_lab)
    print("Accuracy on test values: ", acc[1])


if __name__ == "__main__":
    predict_emotions()
