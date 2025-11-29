from random import randint, choice
from statistics import mean, sqrt

def get_data(dots_count):
    return list([k, k*1.33+randint(1,100)/10] for k in range(100))

batch_szie = 100
epochs = 10000
learning_rate = 0.001
k, bias = 0, 0

data = get_data(10000)

last_loss = 0
for epoch in range(epochs):
    epoch_loss_list = []

    for batch in range(batch_szie):
        dot = choice(data)
        X, Y = dot

        predict = X*k + bias
        loss = (Y-predict)

        epoch_loss_list.append(loss)
    
    
    if mean(epoch_loss_list) > 0:
        k += learning_rate
        bias += learning_rate
    else: 
        k -= learning_rate
        bias -= learning_rate
    
    if epoch % 100 == 0:
        print(f"epoch: {epoch}  avg loss: {mean(epoch_loss_list)}")