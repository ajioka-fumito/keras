from keras.models import model_from_json
from md.loader_basic import loader
import numpy as np
from tqdm import tqdm

gate_model = model_from_json(open("./models/gate/model.json").read())
gate_model.load_weights("./models/gate/weights.h5")

expert_F = model_from_json(open("./models/expert_F/model.json").read())
expert_F.load_weights("./models/expert_F/weights.h5")

expert_P = model_from_json(open("./models/expert_P/model.json").read())
expert_P.load_weights("./models/expert_P/weights.h5")

basic_model = model_from_json(open("./models/basic/model.json").read())
basic_model.load_weights("./models/basic/weights.h5")


x_test,t_test = loader("./data/test")


def predictor1(x):
    gate = gate_model.predict(x)
    gate = np.argmax(gate,axis=1)


    if gate==0:
        predict = expert_F.predict(x)
    else:
        predict = expert_P.predict(x)

    return np.argmax(predict,axis=1)

def predictor2(x):
    gate = gate_model.predict(x)
    predict_F = expert_F.predict(x)
    predict_P = expert_P.predict(x)
    predict = gate[0][0]*predict_F + gate[0][1]*predict_P
    predict = np.argmax(predict,axis=1)
    return predict



cnt = 0
for i,(x,t) in tqdm(enumerate(zip(x_test,t_test))):
    x = np.array([x])
    t = np.argmax(t)
    y = basic_model.predict(x)
    y = np.argmax(y,axis=1)
    if t==y:
        cnt += 1

print("basic",cnt/(i+1))

cnt = 0
for i,(x,t) in tqdm(enumerate(zip(x_test,t_test))):
    x = np.array([x])
    t = np.argmax(t)
    y = predictor1(x)
    if t==y:
        cnt += 1

print("gate",cnt/(i+1))

cnt = 0
for i,(x,t) in tqdm(enumerate(zip(x_test,t_test))):
    x = np.array([x])
    t = np.argmax(t)
    y = predictor2(x)
    if t==y:
        cnt += 1

print("mix_gate",cnt/(i+1))