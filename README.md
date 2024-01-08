# Art Style Classifier
Takes an input piece of art and tries to identify its class of style among 26 choices.

## Outline (needed functionality)

1. Driver file to run the training and inference on the model before loading to a gradio app
2. pre_processing on images
3. batch_loader to prepare batches of data for training splits
4. cnn layers to train with
5. tbd

### Helpful

#### Training Recommendations
* This dataset is pretty large, and pre-loading all of it into datasets/loaders took me an hr+ and still didn't even get to the training loop. I recommend splitting the dataset using dataframes and load that into the sets/loaders to reduce runtime significantly if needed

### Starting Point Pre-Processing Steps
* A resize of 512 strikes a good balance between preserving these details and maintaining manageable file sizes. This scales differently for images and can cause different 2nd dimension values in a (w,h) matrix

#### Activate environment for dependencies
```
python -m venv space
source space/bin/activate
```


##### Next steps
* k-Fold-Cross Validation, needs changes to data loading and training loop (but good for evaluation)
* Next steps (possibly expand to what artists the input style mimics the best, expand on good system design for that, could produce example images of that artists that relate and find common ground with ideas
* Gradio app setup for interface and take input images, expand on possibilities that it offers out of box