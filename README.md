# mdt2022
Master's degree thesis - Detection of Cyberbullying on Social Media

This repository contains the code for model training used in creation of my master's thesis. 
The data the model predicts and learns on consits of users' tweets. The tweets are divided into 6 classes - not cyberbullying, and cyberbullying, which is further classified with types of cyberbullying:
- age
- race
- religion
- sex
- other cyberbullying

There are models created:
- BERT model + MLP
- BERT model + MLP + additional features
- binary BERT model + MLP
- binary BERT model + MLP + additional features

The idea behind having a binary model is to have a model that recognises bullying with a higher certainty making the division between the types of cyberbullying not as important. 
