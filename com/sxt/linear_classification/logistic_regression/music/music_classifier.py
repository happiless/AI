from pprint import pprint

import numpy as np
from scipy.io import wavfile
from scipy import fft
from sklearn.linear_model import LogisticRegression
import pickle


genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
# 读取傅里叶变换之后的数据集，将其做成机器学习所需的X,y
X = []
y = []
for g in genre_list:
    for n in range(100):
        rad = './trainset/'+g+'.'+str(n).zfill(5)+'.fft.npy'
        fft_features = np.load(rad)
        X.append(fft_features)
        y.append(genre_list.index(g))

X = np.array(X)
y = np.array(y)

model = LogisticRegression()
model.fit(X, y)


output = open('model.pkl', 'wb')
pickle.dump(model, output)
output.close()

pkl_file = open('model.pkl', 'rb')
model = pickle.load(pkl_file)
pprint(model)
pkl_file.close()

print('Starting read wavfile')
# music_name = 'heibao-wudizirong-remix.wav'
music_name = 'small-apple.wav'
# music_name = 'xiaobang.wav'

sample_rate, X = wavfile.read('./testset/'+music_name)
print(X.shape)
X = np.reshape(X, (1, -1))[0]
test_fft_features = abs(fft(X)[:1000])
print(sample_rate, test_fft_features,  len(test_fft_features))
result = model.predict([test_fft_features])
print(result)


