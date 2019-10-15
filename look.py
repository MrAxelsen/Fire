import os
import numpy as np
path1 = 'annotations'
path2 = 'images'

filename2 = os.listdir(path2)
filename1 = os.listdir(path1)

filename2 = np.sort(np.array(filename2))
filename1 = np.sort(np.array(filename1))

f = open('train.txt', 'w+')
f.write('')
f.close()
f = open('test.txt', 'w+')
f.write('')
f.close()


#print(filename2)
for i in range(len(filename2)):
    if not '.png' in filename2[i]:
        continue
    if not i%10==0:
        f = open('train.txt', 'a')
        f.write('data/fire/' + filename2[i] + '\n')
        f.close()
    else:
        f = open('test.txt', 'a')
        f.write('data/fire/' + filename2[i] + '\n')
        f.close()
    #pass
