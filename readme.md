# DeepONet for Acoustic Wave Propagation
![Figure_4a_page-0001](https://github.com/user-attachments/assets/33d26ae7-9c45-4bb1-a1b6-bbe4e4806431)

# What is this repository for?
The objective of this repository is to train a DeepONet neural network to model acoustic wave propagation in variable speed models.
# How do I get set up?
1. Download the repository:  
``` git clone https://github.com/Konat23/DeepONet_AWP.git ```  
``` cd DeepONet_AWP ``` 
2. Install environment (requires having conda installed):  
``` conda env create -n DeepONet_AWP -f my_env.yml ```  
``` conda activate DeepONet_AWP ```  
Fix Deepxde bug: go to: ``` ../anaconda3/envs/DeepONet_AWP/lib/python3.9/site-packages/deepxde/data/triple.py ``` and change line 90: ``` self.train_y[indices_branch, indices_trunk] ``` to ``` self.train_y[np.ix_(indices_branch, indices_trunk)] ```
3. Download the Open-Fwi Dataset:  
``` python src/00_get_fwi_models/download_models.py ```
4. Generate the propagations with Finite-difference time-domain:  
``` python src/01_propagator/main.py  ```
5. Preprocess the data:  
``` python src/02_preprocessing/src/main.py ```
6. Train the model:  
``` python src/03_deeponet/src/main.py ```
7. Use the model for inference:  
``` python src/04_model_predict/src/model_predict.py ```
8. Check the results:  
Open and run the Jupyter notebook: ``` src/05_results_review/main_review.ipynb ```
# Quick test (whitout training)
To use a already trained model (example from a paper): 
- Follow steps 1, 2, 3, 4 and 5 above.
- In ``` src/constants ``` replace "V4_exp01_example" with "V4_exp01_paper"
- Inference (step 7) and Check results (step 8)
# Who do I talk to?
Juan Sebastian Guerrero peña  
# Acknowledgments
Thanks are due to the Ministry of Science, Technology and Innovation and the National Hydrocarbons Agency for funding project 205 of 202.
<juan2258400@correo.uis.edu.co>
