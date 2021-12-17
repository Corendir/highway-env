=== 1.Policy Gradient ===
The notebook you have to open for Policy Gradient is Highway - Policy Gradient.ipynb

You have access to pre-trained models by de-commenting the line indicated in the sixth cell.
- model_1 : without ReLU layer (maximise episode length)
- model_2 : with a ReLU layer after Conv2d (minimise loss, most interessing here with PG) <--
- model_3 : with a ReLU layer before Conv2d (maximise episode length)



=== 2.PPO ===
The notebook you have to open for Proximal Policy Optimization is Highway - PPO.ipynb

You have access to pre-trained models by de-commenting the line indicated in the sixth cell.
- model_1 : with a ReLU layer after Conv2d (finally our car is intrepid ! Woohou !!); epochs = 5; batch_size = 1000
- model_2 : same; epochs = 10; batch_size = 2000 <--


=== Notes on the interesing models ===
- Policy gradient model_2 lead to a very careful car.
- PPO model_2 lead to a very intrepid car, it tries to slalom and to gain speed, but it is still clumsy. It deserves to be trained over more epochs.
