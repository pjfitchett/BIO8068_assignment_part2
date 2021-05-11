# Assignment part 2 

library(rinat)
library(sf)
library(keras)

source("download_images.R") 
gb_ll <- readRDS("gb_simple.RDS")

# Bluebell
bluebell_recs <-  get_inat_obs(taxon_name  = "Hyacinthoides non-scripta",
                               bounds = gb_ll,
                               quality = "research",
                               maxresults = 600)

# Foxglove
foxglove_recs <-  get_inat_obs(taxon_name  = "Digitalis purpurea",
                               bounds = gb_ll,
                               quality = "research",
                               maxresults = 600)
# Cowslip
cowslip_recs <-  get_inat_obs(taxon_name  = "Primula veris",
                              bounds = gb_ll,
                              quality = "research",
                              maxresults = 600)

# Download images
download_images(spp_recs = bluebell_recs, spp_folder = "bluebell")
download_images(spp_recs = foxglove_recs, spp_folder = "foxglove")
download_images(spp_recs = cowslip_recs, spp_folder = "cowslip")

# Putting test images into a different folder - 500 for training and validation
# 400 for training, 100 for validation

image_files_path <- "images" 

# list of spp to model - must match folder names
spp_list <- dir(image_files_path) # Automatically pick up names

# number of spp classes - 3 here
output_n <- length(spp_list)

# Create test and species sub-folders
for(folder in 1:output_n){
  dir.create(paste("test", spp_list[folder], sep="/"), recursive=TRUE)
}

# Copy over spp_501.jpg to spp_600.jpg using two loops, deleting the photos
# from the original images folder after the copy
for(folder in 1:output_n){
  for(image in 501:600){
    src_image  <- paste0("images/", spp_list[folder], "/spp_", image, ".jpg")
    dest_image <- paste0("test/"  , spp_list[folder], "/spp_", image, ".jpg")
    file.copy(src_image, dest_image)
    file.remove(src_image)
  }
}

# Training the deep learning model ####

# Need to scale images
img_width <- 150
img_height <- 150
target_size <- c(img_width, img_height)

# Full-colour Red Green Blue = 3 channels
channels <- 3

# Rescale from 255 to between zero and 1
train_data_gen = image_data_generator(
  rescale = 1/255,
  validation_split = 0.2 # define proportion used for validation
)

# Reading all images from a folder
# training images
train_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "training",
                                                    seed = 42)

# validation images
valid_image_array_gen <- flow_images_from_directory(image_files_path, 
                                                    train_data_gen,
                                                    target_size = target_size,
                                                    class_mode = "categorical",
                                                    classes = spp_list,
                                                    subset = "validation",
                                                    seed = 42)

# Check everything has been read in correctly

cat("Number of images per class:")
table(factor(train_image_array_gen$classes))
cat("Class labels vs index mapping")
train_image_array_gen$class_indices

# Display one of the images via as.raster 
plot(as.raster(train_image_array_gen[[1]][[1]][8,,,]))

# Define additional parameters and configure model ####

# number of training samples
train_samples <- train_image_array_gen$n
# number of validation samples
valid_samples <- valid_image_array_gen$n

# define batch size and number of epochs
batch_size <- 32 # define manually
epochs <- 10     # How long to keep training going for

# Define CNN 
# initialise model
model <- keras_model_sequential()

# add layers
model %>%
  layer_conv_2d(filter = 32, kernel_size = c(3,3), input_shape = c(img_width, img_height, channels), activation = "relu") %>%
  
  # Second hidden layer
  layer_conv_2d(filter = 16, kernel_size = c(3,3), activation = "relu") %>%
  
  # Use max pooling
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(0.25) %>%
  
  # Flatten max filtered output into feature vector 
  # and feed into dense layer
  layer_flatten() %>%
  layer_dense(100, activation = "relu") %>%
  layer_dropout(0.5) %>%
  
  # Outputs from dense layer are projected onto output layer
  layer_dense(output_n, activation = "softmax") 

print(model)

# Define error terms and accuracy measures 
# Use categorical crossentropy because more than 2 spp
# Compile the model
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = "accuracy"
)

# Set deep learning model going
# Train the model with fit_generator
history <- model %>% fit_generator(
  # training data
  train_image_array_gen,
  
  # epochs
  steps_per_epoch = as.integer(train_samples / batch_size), 
  epochs = epochs, 
  
  # validation data
  validation_data = valid_image_array_gen,
  validation_steps = as.integer(valid_samples / batch_size),
  
  # print progress
  verbose = 2
)

# Assessing accuracy and loss
plot(history)
# Validation accuracy around 70-75%

# Saving the model for future use ####
# The imager package also has a save.image function, so unload it to
# avoid any confusion
detach("package:imager", unload = TRUE)

# The save.image function saves your whole R workspace
save.image("part2.RData")

# Testing the model ####

# Using the 100 photos in the test folders 
path_test <- "test"

test_data_gen <- image_data_generator(rescale = 1/255)

test_image_array_gen <- flow_images_from_directory(path_test,
                                                   test_data_gen,
                                                   target_size = target_size,
                                                   class_mode = "categorical",
                                                   classes = spp_list,
                                                   shuffle = FALSE, # do not shuffle the images around
                                                   batch_size = 1,  # Only 1 image at a time
                                                   seed = 123)

model %>% evaluate_generator(test_image_array_gen, 
                             steps = test_image_array_gen$n)
# accuracy 77%, loss 61%

# In case of unbalanced data - make a prediction
predictions <- model %>% 
  predict_generator(
    generator = test_image_array_gen,
    steps = test_image_array_gen$n
  ) %>% as.data.frame
colnames(predictions) <- spp_list
predictions
# Shows the probability of each image belonging to each spp

# Create a confusion matrix
# Create 3 x 3 table to store data
confusion <- data.frame(matrix(0, nrow=3, ncol=3), row.names=spp_list)
colnames(confusion) <- spp_list

obs_values <- factor(c(rep(spp_list[1],100),
                       rep(spp_list[2], 100),
                       rep(spp_list[3], 100)))
pred_values <- factor(colnames(predictions)[apply(predictions, 1, which.max)])

library(caret)
conf_mat <- confusionMatrix(data = pred_values, reference = obs_values)
conf_mat
# most sensitive for cowslip, most specific for cowslip
