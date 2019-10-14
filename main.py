import wisardpkg as wp
import pandas as pd
from sklearn.model_selection import KFold

threshold = 100


data = pd.read_csv('./dataset/train.csv', sep=',')
label = data['label'].apply(str)

# print(type(label))

# breakpoint()
test = pd.read_csv('./dataset/test.csv', sep=',')
del data['label']

appliedData = data.apply(lambda x: (x >= threshold).astype(
    int) if x.name != 'label' else x)
testAppliedData = test.apply(lambda x: (x >= threshold).astype(int))


X = appliedData.values.tolist()
y = label.values.tolist()
# breakpoint()
testSet = testAppliedData.values.tolist()
# breakpoint()

addressSize = 3     # number of addressing bits in the ram
ignoreZero = False  # optional; causes the rams to ignore the address 0

# False by default for performance reasons,
# when True, WiSARD prints the progress of train() and classify()
verbose = False

wsd = wp.Wisard(addressSize, ignoreZero=False, verbose=False)

kf = KFold(n_splits=5, shuffle=True)

for k in range(0, 1):
    for train_index, test_index in kf.split(appliedData):
        try:
            wsd.train(X, y)
            out = wsd.classify(testSet)
            breakpoint()
        except Exception as e:
            print(str(e))
