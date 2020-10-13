##################
# part 1:  choose the best model... beta ~ 2 or 4
# part 2:   visualize the "error" components 
# part 3:   create umap / tsne summaries...
# part 4:  create a streamlit app for visualizing the manifold...
# part 4... how to explore the latent space

#%% %%%%%
HOME = "/home/ergonyc"    #linux

ROOT_DIR = HOME + "/Projects/Project2.0/SneakerGen/beta-vae"

IMGRUN_DIR = ROOT_DIR + "/imgruns"
TXTRUN_DIR = ROOT_DIR + "/txtruns"

img_run_id = "1001-1510"   
cf_img_size = 224  
cf_kl_weight = 1.0
txt_run_id = "1001-1830"  

params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,cf_img_size,cf_kl_weight]))

#%%  %%%%%%%%%

# load the data files
def load_models_and_loss(p ):
    """
    load the models and loss so we can find the best fit.. 

    input: params = dict (root_dir, img_run_id, txt_run_id, img_size,kl_weight, ..)
    """


# train_data = np.load(os.path.join(cf.DATA_DIR, 'train_data.npy'), allow_pickle=True)
# val_data = np.load(os.path.join(cf.DATA_DIR, 'val_data.npy'), allow_pickle=True)
# all_data = np.load(os.path.join(cf.DATA_DIR, 'all_data.npy'), allow_pickle=True)

ut.dump_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"),curr_losses)
ut.dump_pickle(os.path.join(lg.root_dir,"snk2vec.pkl"), snk2vec)
ut.dump_pickle(os.path.join(lg.root_dir,"snk2loss.pkl"), snk2loss)



# text
    print(f"loading train/validate data from {cf.DATA_DIR}")
    train_data = np.load(os.path.join(cf.DATA_DIR, 'train_txt_data.npy'), allow_pickle=True)
    val_data = np.load(os.path.join(cf.DATA_DIR, 'val_txt_data.npy'), allow_pickle=True)
    all_data = np.load(os.path.join(cf.DATA_DIR, 'all_txt_data.npy'), allow_pickle=True)
id_label = np.load(os.path.join(cf.DATA_DIR,'mnp.npy'))  #IDs (filenames)
descriptions = np.load(os.path.join(cf.DATA_DIR,'dnp.npy')) #description
description_vectors = np.load(os.path.join(cf.DATA_DIR,'vnp.npy')) #vectors encoded
padded_encoded_vector = np.load(os.path.join(cf.DATA_DIR,'pnp.npy')) #padded encoded

ut.dump_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"),curr_losses)




img_run_id = "1001-2040"   
cf_img_size = 224  
cf_kl_weight = 0.5
#txt_run_id = "1001-2155"  


img_run_id = "1002-0844"   
cf_img_size = 224  
cf_kl_weight = 2.0
# 1002-0957

img_run_id = "1002-1244"   
cf_img_size = 224  
cf_kl_weight = 4.0
# 1002-1634

img_run_id = "1002-1833"   
cf_img_size = 224  
cf_kl_weight = 8.0
# 1002-2117 

img_run_id = "1002-2316"   
cf_img_size = 224  
cf_kl_weight = 16.0
# 1003-0847 

img_run_id = "1003-1046"   
cf_img_size = 224  
cf_kl_weight = 0.25
# 1003-1159 


img_run_id = "1004-1938"  
cf_img_size = 224 
cf_kl_weight = 32.
# 1004-2334 

#img_run_id = "1005-1010" # txtmodel probably run with against 1004-1938 
#cf_img_size = 224 
#cf_kl_weight = 2.
# 1005-2334 


img_run_id = "1007-1709"  
cf_img_size = 224  
cf_kl_weight = 2.0  # with more cvae epichs... (overfitting)
# 1007-1858


