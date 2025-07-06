#####################################################
############ READ FIRST: IMPORTANT NOTE #############
#####################################################

# NOTE: this script involves training a neural network model. Model
# training is resource-intensive. In the context leverage project,
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

# In the context leverage project, this script was run 4 times 
# to fit 4 randomly initialized instances of the same RNN model.
# Each time the script is run, the fitted model is saved in the 
# models subdirectory.

#####################################################
#################### DESCRIPTION ####################
#####################################################

# Measures of context leverage capture the degree to which words
# occur in similar contexts to words similar in meaning that are
# learned earlier in development. 

# The focus of this script is to train an RNN on the CHILDES corpus.
# Training involves forming representations of words based on the language
# contexts in which they occur. Representations will be used co calculate
# measures of context leverage in another script.


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

#################################################
############## WORKING DIRECTORY ################
#################################################

setwd("~/WORKING DIRECTORY")

#################################################
############# CORPUS PREPROCESSING ##############
#################################################

# Load CHILDES corpus
childes_text <- readRDS(url("https://language-corpus-childes-english.s3.us-east-2.amazonaws.com/childes_lemmatized_stop.rds"))


# Order corpora by age
childes_text <- childes_text %>%
  dplyr::arrange(target_child_age)

# Do additional preprocessing steps to replace contractions with
# their non-contraction equivalents

# Replace "'ll" with "will"
childes_text <- childes_text %>% 
  dplyr::mutate_if(is.character, str_replace_all, pattern = "'ll", replacement = ' will')

# Expand contractions
data(contractions)

contraction_key <- contractions 

extra_contractions <- data.frame(contraction = c("i'm", "i'll", "i've", "i'd", "that'd", "where'd", 
                                                 "where're", "what're", "willn't", "there're",
                                                 "whyn't", "why'd", "haven't", "o'clock",
                                                 "who're", "hadn't", "d'you", "d'ya", "ya'll", "y'all",
                                                 "c'mon", "how're", "c'mere", "why're", "come'ere", "it'd",
                                                 "she'd", "he'd", "what've", "why've", "where've",
                                                 "s'more", "y'know", "y'want", "cann't", "come'ere",
                                                 "com'ere"),
                                 expanded = c("i am", "i will", "i have", "i would", "that would", "where did", 
                                              "where are", "what are", "will not", "there are",
                                              "why not", "why did", "have not", "oclock",
                                              "who are", "had not", "do you", "do you", "you all", "you all",
                                              "come on", "how are", "come here", "why are", "come here", "it would",
                                              "she would", "he would", "what have", "why have", "where have",
                                              "some more", "you know", "you want", "can not", "come here", "come here"))

contraction_key <- rbind(contraction_key, extra_contractions)

childes_unnested <- childes_text %>%
  # Convert each transcript into long format where each row is a word in the transcript
  tidytext::unnest_tokens(word, Text) %>%
  left_join(contraction_key, by = c("word" = "contraction"))

childes_unnested$word <- ifelse(is.na(childes_unnested$expanded), childes_unnested$word, childes_unnested$expanded)

childes_unnested <- childes_unnested %>%
  # Convert each transcript into long format where each row is a word in the transcript
  tidytext::unnest_tokens(word, word) %>%
  dplyr::select(id, target_child_age, word)


# Get word frequencies and log word frequencies
childes_freq <- childes_unnested %>%
  # Count the number of time each word occurs and log transform
  dplyr::count(word, name = "freq_childes") %>%
  dplyr::arrange(desc(freq_childes))

#################################################
################ TOKENIZE WORDS #################
#################################################

cutoff_value <- 4096

# Convert lower freq into "UNK"
childes_freq$freq_cutoff <- c(rep(1, cutoff_value), rep(0, nrow(childes_freq)-cutoff_value))

childes_unnested <- left_join(childes_unnested, childes_freq)

childes_unnested$word_token <- ifelse(childes_unnested$freq_cutoff == 1, childes_unnested$word, "UNK")

childes_text <- childes_unnested %>%
  group_by(target_child_age, id) %>%
  dplyr::summarise(Text = paste(word_token, collapse=" "))

# Divide into sliding windows
window_size = 11


childes_windows <- childes_text %>%
  tidytext::unnest_tokens(ngram, Text, token = "ngrams", n = window_size)

childes_windows <- na.omit(childes_windows)


# Divide windows into consecutive bins
num_bins <- 40
childes_windows$age_bin <- dplyr::ntile(childes_windows$target_child_age, num_bins)

# Calculate batch size based on number of samples per bin
batches_per_bin <- 10
batch_size <- round(nrow(childes_windows) / (num_bins * batches_per_bin))

# Generate names for files containing the tokenizer and tokenized text
tokenized_name <- paste("tokenizer/CHILDES_tokenized_win", window_size, ".rds", sep = "")
tokenizer_name <- paste("tokenizer/CHILDES_tokenizer_win", window_size, sep = "")

# Check whether you have already saved the tokenizer / tokenized text
# by checking whether there are files in the tokenizer folder.
# If there are no files in the folder, create the tokenizer / tokenized 
# text and save them in the tokenizer folder. If there are files in the folder, 
# read them.
tokenize_files <- list.files(path = "tokenizer")

if(length(tokenize_files) == 0) {
  
  # Keras tokenization/words to integers
  max_features <- cutoff_value + 10
  
  tokenizer <- text_tokenizer(num_words = max_features) %>%
    fit_text_tokenizer(childes_windows$ngram)
  
  text_seqs <- texts_to_sequences(tokenizer, childes_windows$ngram)
  
  vocab_size = length(tokenizer$word_index) + 1
  
  saveRDS(text_seqs, tokenized_name)
  
  save_text_tokenizer(tokenizer, tokenizer_name)
} else {
  tokenizer <- load_text_tokenizer(tokenizer_name)
  text_seqs <- readRDS(tokenized_name)
}


vocab_size = length(tokenizer$word_index) + 1


#################################################
############## TRAINING PARAMETERS ##############
#################################################

seq_length = window_size - 1

# If embedding into low-dimensional space, choose value
# that is a factor of 10 smaller than vocab_size 
embedding_size <- round(vocab_size/10)

#################################################
############ MODEL TRAINING: HUEBNER ############
#################################################




# Define model. 
# Note: This model has minor adaptations from description in Huebner
# (1) Adds a masking layer that masks zeros in the input.
#     The tokenizer numbers words starting from 1, so masking zeros allows us to test
#     the model later with input containing just a single word and the others masked)
# (2) Does the one-hot encoding of input using an embedding layer in the model. One-hot 
#     encoding the input outside the model was too resource-intensive. Similarly, to 
#     avoid one-hot encoding the output, use sparse_categorical_crossentropy loss

huebner_model <- keras_model_sequential()

huebner_model <- huebner_model %>%
  # Mask zeros in input 
  layer_masking(input_shape = seq_length) %>%
  
  # Embedding Option 1: onehot embedding (as in Huebner)
  layer_embedding(name = "Onehot_layer", input_dim = vocab_size, output_dim = vocab_size, input_length = seq_length,
                  embeddings_initializer = "identity", trainable = FALSE)  %>%
  
  # Embedding Option 2: trainable lower-dimensional embedding
  # layer_embedding(name = "Embedding_layer", input_dim = vocab_size, output_dim = embedding_size, input_length = seq_length)  %>%
  
  # Embedding Option 3: non-trainable lower-dimensional embedding
  # layer_embedding(name = "Embedding_layer", input_dim = vocab_size, output_dim = embedding_size, input_length = seq_length, trainable = FALSE)  %>%
  
  # Dropout layer (optional)
  # layer_dropout(rate = .5) %>%
  
  # Hidden/Output Option 1: Hidden layer with prediction of final word from preceding (as in Huebner)
layer_simple_rnn(name = "Hidden1", units = 512, activation = 'tanh') %>% #tanh in Huebner, alternative relu
  layer_dense(units = vocab_size, activation = 'softmax')

# Hidden/Output Option 2: Hidden layer with return sequences + output
# layer_simple_rnn(name = "Hidden1", units = 512, activation = 'relu', return_sequences = TRUE) %>% 
# time_distributed(layer_dense(units = vocab_size, activation = 'softmax'))

# Compile model (not currently splitting into training & validation; focus
# here is on testing semantic organization)
huebner_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = optimizer_adagrad(learning_rate = 0.01), # Used in Huebner
  #optimizer = 'adam', # Alternative common approach
  metrics = c('accuracy')
)

# If you would like to estimate how long the full training will take,
# set the number of bins to just 2 by uncommenting the line below. 
# The total number is 40, so the total training will take about 20 times as long.
# num_bins <- 2

num_training_rounds <- 1

activation_name = "tanh"
optimizer_name = "adagrad"
encoding = "onehot"

  
for(t in 1:num_training_rounds){
  
  # Generate a random ID for the model
  model_id <- paste(c(sample(LETTERS[1:26], 3, T), sample(0:9, 3, T)), collapse = "")
  
  # Train model consecutively on each age bin
  training_history <- c()
  
  model_name <- paste("models/model_huebner_win", window_size, "_", activation_name, "_", optimizer_name, "_", encoding, "_", "_model_id", model_id, "_", "training_round", t, ".h5", sep = "")
  history_name <- paste("models/history/model_huebner_history_win", window_size, "_", activation_name, "_", optimizer_name, "_", encoding, "_", "_model_id", model_id, "_", "training_round", t, ".h5", sep = "")
  
  for(i in 1:num_bins) {
    
    print(paste("bin = ", i, sep = ""))
    
    # Get the indeces of the windows within the bin
    target_windows <- which(childes_windows$age_bin == i )
    
    # Get the tokenized windows, format in a matrix where each row is a window
    # and each column is an integer corresponding to a word
    text_matrix <- matrix(unlist(text_seqs[target_windows]), ncol = window_size, byrow = T)
    
    # Split into input and output
    
    # Input (first word through penultimate word)
    text_x <- text_matrix[ , 1: (ncol(text_matrix) - 1)]
    
    # Output if only predicting final word
    text_y <- matrix(text_matrix[ , ncol(text_matrix)], ncol = 1)
    
    # Output if using return_sequences & timedistributed (second word through final word):
    # text_y <- text_matrix[ , 2: ncol(text_matrix)]
    
    # fit model
    temp_history <- huebner_model %>% fit(
      text_x, text_y,
      #batch_size = batch_size,
      batch_size = 64,
      #shuffle = FALSE,
      epochs = 4)
    
    
    save_model_hdf5(huebner_model, model_name)
    
    training_history <- c(training_history, temp_history)
    saveRDS(training_history, file = history_name)
  }
  
}
  
