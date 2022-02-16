import numpy as np
import matplotlib.pyplot as plt

def exp_decay_learning_rate(init_lr, step, decay_steps=1e5, decay_rate=0.96):
    return init_lr * decay_rate ** (step / decay_steps)

def lin_decay_learning_rate(init_lr, step, end_lr=0.0001, decay_steps=1e5, decay_rate=0.04):
    return (1 - decay_rate) * init_lr + decay_rate * end_lr 

init = 0.001
lr1 = [init]
lr2 = [init]
steps = 1000

current_lr1 = init
current_lr2 = init
for step in range(steps):
    current_lr1 = exp_decay_learning_rate(current_lr1, step + 1, decay_steps=steps)
    lr1.append(current_lr1)


    current_lr2 = lin_decay_learning_rate(current_lr2, step + 1, decay_steps=steps, end_lr=0.1*init, decay_rate=0.004)
    lr2.append(current_lr2)

fig, ax = plt.subplots(2, figsize=(15, 8))
ax[0].plot(lr1)
ax[0].set_title("Exponential Decay")
ax[0].set_ylabel("Learning Rate")
ax[0].set_xlabel("Epochs")

ax[1].plot(lr2)
ax[1].set_title("Linear Decay")
ax[1].set_ylabel("Learning Rate")
ax[1].set_xlabel("Epochs")
plt.show()

