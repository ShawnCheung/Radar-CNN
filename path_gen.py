import os
import numpy as np
paths = os.listdir("./train-c")
select = np.random.random(len(paths))
train = open("./train_path.txt", 'w')
test = open("./test_path.txt",'w')

for idx, x in enumerate(select):
    if x<=0.75:
        train.writelines(paths[idx]+'\n')
    else:
        test.writelines(paths[idx]+'\n')
train.close()
test.close()
print("finish")