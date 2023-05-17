from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def relu(x):
    return np.maximum(0, x)

x = np.array([0, 1, 2, 3])
y = x * 2 + 1
print("x: ", x)
print("y: ", y)

model = Sequential()
model.add(Dense(10, input_shape=(1,), activation='relu')) # 첫 번째 은닉층 (10개의 유닛)
model.add(Dense(10, activation='relu')) # 두 번째 은닉층 (10개의 유닛)
model.add(Dense(10, activation='relu')) # 세 번째 은닉층 (10개의 유닛)

model.add(Dense(1, activation='linear')) # 선형 활성화 함수를 사용한 출력층
model.compile('SGD', 'mse')

model.fit(x, y, epochs=100, verbose=1)