from sim import add_real_noise
import numpy as np
import time
start = time.time()
add_real_noise(np.random.randn(31, 200), 2, durOfTrial=2)
end = time.time()
print(end-start)