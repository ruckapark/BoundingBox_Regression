# Simple Conv Net model - bounding box regression around lines

> This is part of a project for the development of software for animal detection in a more complicated image environment.
> This is the first test and possible base for using such a method for animal tracking in real time with more simple methods than YOLO etc.

- writefile is not useful - it was used in the preprocessing to have ordered data.

## Instructions for user

- Clone repo
- **Generate dataset** with image_gen.py:
	- Create two directories (TestSet and ValidationSet)
	> I choose here to create the sets without splitting in the code
	
	- Modify paths in code to personal adress and select dataset size
	```python
	os.chdir('ValidationSet') CHANGE
    dataset_size = 1000 CHANGE
	```
	
	- Run to generate images for training purposes

- Activate tf environment if necessary
- Run training in jupyter notebook

## CNN architecture

Very simple architecture, with scaling performed pre-entry to NN.
This will be made more sophisticated as necessary as image complexity increases.

Author : George Ruck