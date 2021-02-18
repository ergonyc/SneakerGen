# ~~Sneaker Generative Design~~
Andy Henrie

## this project is currently inactive and is being refactored to a fastai API based project

### Project Goal:  
Generate Sneaker pictures from examples and text descriptions to increase the speed and efficacy of the design process.


# Repo Usage
WIP:

 1. Download all data and update configuration file according to download locations in 'configs.py'.
 
      a. IMPORTANT NOTE: Download dabase... INSTRUCTIONS TBD
      
 2. Train shape encoder model using 'cvae.py' file. Confirm model performance using provided visualization methods.
 3. Gather data and generate descriptions for objects. WIP

 4. Train text encoder model using 'text2sneak.pp'. Confirm model performance using provided visualization methods.
 5. Create UMAP plots using **umap.py** which generates a pandas csv file with relevant info for use in the streamlit app.
 6. WIP **streamlit_app.py** file with Streamlit.

      
The following files are used by the main programs:
- **utils.py** Contains many commonly useful algorithms for a variety of tasks.
- **textspacy.py** Class that contains the text encoder model.
- **cvae.py** Class that contains the shape autoencoder model.
- **logger.py** Class for easily organizing training information like logs, model checkpoints, plots, and configuration info.
- **easy_tf2_log.py** Easy tensorboard logging file from [here](https://github.com/mrahtz/easy-tf-log) modified for use with tensorflow 2.0

# ~~Streamlit App ~~
WIP  
            
## Available Tabs:            
- ### Text to shape generator
- ### Latent vector exploration
- ### Shape interpolation

## Text to Sneak Generator
Initial tests proved proof of concept, but better formed "disentangled" latent spaces are nescessary to map the text descriptions usefully to the latents for generation.

## Latent Vector Exploration
The shape embedding vectors reduced from the full model dimensionality of 128 dimensions
down to 2 so they can be viewed easily. The method for dimensionality reduction was UMAP (and TSNE)


## Sneaker Interpolation
This is intended to show how well the modelcan interpolate between various sneaker models. 

This is essentially a constrained random walk between two places in the latent space via the db examples along the way.
To generate these plots, the algorithm finds the nearest K shape embedding vectors and randomly picks one of them.
Then it interpolates between the current vector and the random new vector and at every interpolated point it generates a new model from the interpolated latent space vector. Then it repeats to find new vectors.
