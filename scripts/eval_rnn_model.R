#####################################################
############ READ FIRST: IMPORTANT NOTE #############
#####################################################

# NOTE: this script is intended to run AFTER the train_rnn_model script
# which fits an RNN model. 
# Model training is resource-intensive. In the context leverage project,
# models were trained using supercomputing resources. This script
# is being shared for the purpose of reproducibility and peer review,
# but it is advised that anyone attempting to run the script should
# arrange to do so using supercomputing resources, which may involve
# setting up a system of directories containing files needed in the
# script.

# To run, you should set up a directory with the following subdirectories:
# |-- data
# |   |-- aoa_mcdi_english.csv
# |-- tokenizer
# |-- models
# |   |-- history
# |-- similarity_matrices

# This script should be run after the train_rnn_model script has been run
# at least once. In the context leverage project, it was run 4 times,
# resulting in 4 fitted instances of the same RNN model trained on the 
# CHILDES corpus.

#####################################################
#################### DESCRIPTION ####################
#####################################################

# Measures of context leverage capture the degree to which words
# occur in similar contexts to words similar in meaning that are
# learned earlier in development. 

# The focus of this script is to extract the similarities between
# representations of words from RNNs trained RNN on the CHILDES corpus.
# These similarities will be used to calculate measures of context leverage 
# in the context_leverage script.

#####################################################
################### LOAD PACKAGES ###################
#####################################################


library(dplyr)
library(tidyr)

library(stringr)
library(tidytext)

library(qdapDictionaries)

library(keras)
library(tensorflow)

library(proxyC)

#################################################
############## WORKING DIRECTORY ################
#################################################

setwd("~/WORKING DIRECTORY")

#################################################
############ READ TOKENIZER & WORDS #############
#################################################

# Read in tokenizer and generate integer-word key
tokenizer <- load_text_tokenizer("tokenizer/CHILDES_tokenizer_win11")
integer_key <- data.frame(word = names(tokenizer$word_index),
                          integer = unlist(tokenizer$word_index))

# Read in tokenized text
text_seqs <- readRDS("tokenizer/CHILDES_tokenized_win11.rds")

# Read in words included in the MCDI
mcdi <- read.csv("data/aoa_mcdi_english.csv")


# Identify the token integers corresponding to the MCDI words
probe_key <- integer_key[integer_key$word %in% unique(mcdi$word),]

# Generate an empty matrix that will be used to store the vectors
# corresponding to each of the MCDI words
representations <- matrix(NA, nrow = nrow(probe_key), ncol = 512)

#################################################
################ WORD SIMILARITY ################
#################################################

# Get the names of all the fitted models in the models directory
models <- list.files(path = "models/", pattern = "huebner.*round1", full.names = T)

# For each model, get the vector representations of all the MCDI words, then
# calculate cosine similarity between them
for(mod in 1:length(models)) {
  
  # Read in model(s)
  model <- load_model_hdf5(models[mod])
  
  # Get the name of the model and set up a file in which to save the similarities
  model_name <- gsub("\\.h5|models//model_", "", models[mod])
  similarities_name <- paste("similarity_matrices/", model_name, "_sim_matrix.rds", sep = "")
  
  # Generate an "activation" version of the model in which the "output" is the
  # internal representation layer. 
  activation_model <- keras_model(inputs = model$input, outputs = get_layer(model, "Hidden1")$output)
  
  # Window size
  window_size <- 11
  
  # Feed in each MCDI word as input to the model. The model was
  # trained on input consisting of 11-word windows, so here we format
  # the input as just the word combined with 10 0s that will be masked
  # and therefore not affect the output of the activation model
  for(i in 1:nrow(probe_key)) {
    
    # Get the integer corresponding to the word
    integer_input <- probe_key$integer[i]
    print(probe_key$word[i])
    
    # Format the input
    model_input <- matrix( c(integer_input, rep(0, window_size - 2) ), nrow = 1)
    
    # Get the output of the activation model for the word - this is the word's
    # vector representation
    output <- predict(activation_model, model_input)
    
    # Get just the vector
    output <- output[1, 1:512]
    
    # Store this in the representations matrix
    representations[i, ] <- output
  }
  
  # Calculate cosine similarities between all words. This
  # results in a similarity matrix between all word pairs
  word_similarities <- proxyC::simil(representations, margin = 1, method = "cosine")
  word_similarities <- as.matrix(word_similarities)
  
  # Name the rows and columns with the words
  colnames(word_similarities) <- probe_key$word
  rownames(word_similarities) <- probe_key$word
  
  # Save the similarity matrix.
  saveRDS(word_similarities, file = similarities_name)
  

}

