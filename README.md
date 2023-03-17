# Metspec

This code performs the detection and photometry of slitless meteor spectra captured by DSLR. The goal is to recognize the track of bright stars and Na and Mg spectra and perform photometry on it. It is not completed, and more adjustments are needed. 

![image-20230309221416408](fig/image-20230309221416408.png)

![image-20230317005926933](fig/image-20230317005926933.png)

## Note

* `generate_dataset.ipynb`: generates the fake meteor dataset to train the network
* `training.ipynb`: trains the network
* `photometry.ipynb`: performs photometry based on the result of network
* `utils.py`: pre-defined utility functions to prettify the code

