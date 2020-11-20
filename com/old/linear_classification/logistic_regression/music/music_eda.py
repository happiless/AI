from scipy import fft
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import numpy as np
'''
sample_rate, X = wavfile.read('/Users/zhanghaibin/Downloads/线性分类算法/02softmax回归/01/'
                              'genres/blues/converted/blues.00000.au.wav')
print(sample_rate, X.shape)

plt.figure(figsize=(10, 4), dpi=80)
plt.xlabel('time')
plt.ylabel('frequency')
plt.grid(True, linestyle='-', color='0.75')
specgram(X, Fs=sample_rate, xextent=(0, 30))
'''


def plotSpec(g, n):
    (sample_rate, X) = wavfile.read('/Users/zhanghaibin/Downloads/线性分类算法/02softmax回归/01/'
                                    'genres/' + g + '/converted/' + g + '.' + n + '.au.wav')
    specgram(X, Fs=sample_rate, xextent=(0, 30))
    plt.title(g+'-'+n[-1])


plt.figure(figsize=(9, 9), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(3, 3, 1); plotSpec('classical', '00000')
plt.subplot(3, 3, 2); plotSpec('country', '00000')
plt.subplot(3, 3, 3); plotSpec('disco', '00000')
plt.subplot(3, 3, 4); plotSpec('hiphop', '00000')
plt.subplot(3, 3, 5); plotSpec('jazz', '00000')
plt.subplot(3, 3, 6); plotSpec('metal', '00000')
plt.subplot(3, 3, 7); plotSpec('pop', '00000')
plt.subplot(3, 3, 8); plotSpec('reggae', '00000')
plt.subplot(3, 3, 9); plotSpec('rock', '00000')
plt.tight_layout(pad=0.4, w_pad=0, h_pad=1)
plt.show()


sample_rate, X = wavfile.read('/Users/zhanghaibin/Downloads/线性分类算法/02softmax回归/01'
                              '/genres/blues/converted/blues.00000.au.wav')
plt.figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(2, 1, 1)
plt.xlabel('time')
plt.ylabel('frequency')
specgram(X, Fs=sample_rate, xextent=(0, 30))
plt.subplot(2, 1, 2)
plt.xlabel('frequency')
plt.xlim((0, 3000))
plt.ylabel('amp')
plt.plot(fft(X, sample_rate))
plt.show()


# 准备音乐数据，把音乐文件一个个的去使用傅里叶变换，并且把傅里叶变换之后的结果落地保存
# 提取特征
def create_fft(g, n):
    rad = '/Users/zhanghaibin/Downloads/线性分类算法/02softmax回归/01/genres/'+g+'/converted/'+g+'.'+str(n).zfill(5)+'.au.wav'
    sample_rate, X = wavfile.read(rad)
    fft_features = abs(fft(X)[:1000])
    sad = './trainset/'+g+'.'+str(n).zfill(5)+'.fft'
    np.save(sad, fft_features)


genre_list = genre_list = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal']
for g in genre_list:
    for n in range(100):
        create_fft(g, n)
