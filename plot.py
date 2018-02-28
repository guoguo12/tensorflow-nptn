import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Usage: python {} [log]'.format(sys.argv[0]))
    sys.exit(1)
path = sys.argv[1]

epoch = []
val = []
for line in open(path):
    match_obj = re.search(r'epoch=(.*?),.*val=(.*?),', line)
    if match_obj:
        epoch.append(float(match_obj.group(1)))
        val.append(float(match_obj.group(2)))

assert len(epoch) == len(val)
print('{} lines of log found'.format(len(epoch)))

plt.plot(epoch, val, 'g-')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.savefig('validation_accuracy.png')
