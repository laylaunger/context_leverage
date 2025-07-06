#####################################################
#################### DESCRIPTION ####################
#####################################################

# Measures of context leverage capture the degree to which words
# occur in smilar contexts to words similar in meaning that are
# learned earlier in development. 

# The focus of this script is to supplement the context_leverage script
# by testing whether results are consistent across different parameter
# choices for the context overlap metric. 

# Prior to running this script, you should ensure that you
# have the data folder, which includes: 
# (1) a file containing utterance statistics: utt_stats_english.rds
# (2) a file containing age of acquisition data: aoa_mcdi_english.csv
# (3) a file containing a list of common first names that 
#     will be replaced with a common token: childes_names.txt
# (4) a file with concreteness ratings: abcon_mcdi_english.csv
# (5) a file linking words to semantic categories: categories_english.csv



#####################################################
################### LOAD PACKAGES ###################
#####################################################

# Install any packages not yet installed

library(plyr)
library(tidyverse)
library(widyr)
library(tidytext)
library(purrr)

library(readxl)
library(stringr)

library(ggplot2)
library(ggpubr)
library(cowplot)
library(ggstatsplot)
library(paletteer)


library(textstem)
library(textdata)
library(text2vec)
library(wordspace)
library(lemmar)
library(qdapDictionaries)

library(ppcor)
library(scam)


#####################################################
################## LOAD MCDI DATA ###################
#####################################################

# Load age of acquisition data (calculated from MCDI data)
mcdi_aoa <- read.csv("data/aoa_mcdi_english.csv", fileEncoding="latin1")

#####################################################
################# LOAD ABCON DATA ###################
#####################################################

# Load ratings of word concreteness
abcon <- read.csv("data/abcon_mcdi_english.csv", fileEncoding="latin1")

#####################################################
############## LOAD UTTERANCE DATA ##################
#####################################################

# Load stats that indicate the mean length of utterances in which words occur
utt_stats <- readRDS(file = "data/utt_stats_english.rds")

#####################################################
#################### GET CORPUS #####################
#####################################################

# Load the preprocessed CHILDES corpus
childes_text <- readRDS(url("https://language-corpus-childes-english.s3.us-east-2.amazonaws.com/childes_lemmatized_stop.rds"))

#####################################################
################# GET COMMON NAMES ##################
#####################################################

# Load a list of common proper names in the CHILDES corpus
names_file <- list.files(pattern = "names", recursive = TRUE, full.names = TRUE)
names_to_replace <- readLines(names_file)


#####################################################
################## GET CATEGORIES ###################
#####################################################

# Load key that maps a set of words to semantic categories 
categories_raw <- read.csv("data/categories_english.csv", fileEncoding="latin1")


#####################################################
########## LOAD CHILDES & CALC FREQUENCY ############
#####################################################

# First, load file containing corpus text
# This will load a dataframe in which each row is a full CHILDES 
# transcript. Each transcript is associated with a unique id, the 
# id of the corpus, and the age of the target child.
# The corpus has been preprocessed by: 
# (1) removing punctuation aside from apostrophes in contractions
# (2) removing utterances that did not come from a caregiver
# (3) lemmatizing


# Do additional preprocessing steps to replace contractions with
# their non-contraction equivalents

# Replace "'ll" with "will"
childes_text <- childes_text %>% 
  dplyr::mutate_if(is.character, str_replace_all, pattern = "'ll", replacement = ' will')

# Get common contractions and their expanded equivalents
# from qdapDictionaries
data(contractions)

contraction_key <- contractions 

# Add some additional contractions
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

contraction_key <- rbind(contraction_key, extra_contractions) %>%
  dplyr::distinct()

# Convert each transcript into long format where each row is a word in the transcript
childes_unnested <- childes_text %>%
  tidytext::unnest_tokens(word, Text) 

# Add the contraction key to the unnested corpus
childes_unnested <- left_join(childes_unnested, contraction_key, by = c("word" = "contraction"))

# Replace contractions with their expanded forms
childes_unnested$word <- ifelse(is.na(childes_unnested$expanded), childes_unnested$word, childes_unnested$expanded)

# Unnest again to separate out the expanded forms of contractions
childes_unnested <- childes_unnested %>%
  tidytext::unnest_tokens(word, word) %>%
  dplyr::select(id, corpus_id, target_child_age, word)


# Next, replace people's names with a common token
childes_unnested$word <- ifelse(childes_unnested$word %in% names_to_replace, "pname", childes_unnested$word)

# Get word frequencies and log word frequencies,
# then arrange in descending order of frequency
childes_freq <- childes_unnested %>%
  # Count the number of time each word occurs and log transform
  dplyr::count(word, name = "freq") %>%
  dplyr::mutate(log_freq = log(freq)) %>%
  dplyr::arrange(desc(freq))


# We will get only the top N most frequent words, and replace
# other words with an "UNK" token. Here, define the value of
# N as the cutoff_value
cutoff_value <- 4096

# For each word in childes_freq, identify whether it is above 
# the cutoff value or below (1 for words above, 0 for words 
# below)
childes_freq$freq_cutoff <- c(rep(1, cutoff_value), rep(0, nrow(childes_freq)-cutoff_value))

# Add frequency information to unnested corpus
childes_unnested <- left_join(childes_unnested, childes_freq)

# Replace lower freq with "UNK"
childes_unnested$word <- ifelse(childes_unnested$freq_cutoff == 1, childes_unnested$word, "UNK")


# Re-nest the words in the transcripts back together 
childes_text <- childes_unnested %>%
  group_by(id, corpus_id, target_child_age) %>%
  dplyr::summarise(Text = paste(word, collapse=" ")) %>%
  dplyr::ungroup()

#####################################################
############### COMBINE CONTROL STATS ###############
#####################################################

# Combine all the "control" statistics that will be used 
# to predict word learning in addition to context leverage
# metrics (calculated below)
control_stats <- childes_freq %>%
  dplyr::filter(word %in% unique(mcdi_aoa$word)) %>%
  dplyr::select(c(word, log_freq)) %>%
  left_join(abcon) %>%
  left_join(utt_stats)


#####################################################
################# LOAD CATEGORIES ###################
#####################################################

# Convert names of categories to lower case
categories_raw <- mutate_all(categories_raw, .funs=tolower)

# Separate lists of words in each category into separate columns
categories_raw <- categories_raw %>% separate_longer_delim(word, delim = ", ")

# Remove any reduncancies
categories_raw <- categories_raw %>%
  dplyr::distinct()


# Lemmatize the words in the categories so that they
# match the lemmatized words in the preprocessed corpus

# Get language lemmas
language_lemma <- "hash_lemma_en"
data(list = language_lemma)

lemmas <- get("hash_lemma_en")

categories_raw$lemmatized <- lemmar::lemmatize_words(categories_raw$word, dictionary = lemmas)

# Only keep words that appear in the CHILDES corpus
categories_filter <- categories_raw[categories_raw$lemmatized %in% childes_freq$word,]

# Generate each possible pair of words and their respective categories
# First, add new column that pastes together a word with its category
categories_filter$word_category <- paste(categories_filter$lemmatized, categories_filter$category, sep = "_")

# Identify number of items in each category
categories_filter <- categories_filter %>%
  group_by(category) %>%
  dplyr::mutate(num_words = length(unique(lemmatized)))


# Second, generate each pair of pasted word_category values
pairs <- data.frame(t(combn(categories_filter$word_category, 2)))

# Third, separate pasted word_category values to generate a dataframe with
# a column for each word in the pair and the category of each word in the pair
pairs <- setNames(data.frame(str_split_fixed(pairs$X1, "_", 2), str_split_fixed(pairs$X2, "_", 2)),
                  c("word1", "category1", "word2", "category2"))

# Add a column that indicates whether words in a pair belong to the same or
# different categories
pairs$type <- ifelse(pairs$category1 == pairs$category2, "same", "diff")

# Add a column that indicates the category membership of both words in a pair
# if they belong to the same category, contains the value "diff" if they belong to 
# different categories
pairs$category <- ifelse(pairs$type == "same", pairs$category1, "diff")

pairs_invert <- pairs[,c("word2", "category2", "word1", "category1", "type", "category")]
names(pairs_invert) <- names(pairs)

pairs <- rbind(pairs, pairs_invert)

# Only words with mcdi data
pairs <- pairs[pairs$word1 %in% mcdi_aoa$word & pairs$word2 %in% mcdi_aoa$word, ]

# Remove any duplicate rows
pairs <- dplyr::distinct(pairs)


# If a word was in multiple categories so that it ends up having both
# same and different category entries for the same paired word, just 
# keep the "same"
pairs <- pairs %>%
  group_by(word1, word2) %>%
  dplyr::mutate(double_entry = ifelse(  length(type) == 2, 1, 0        )) %>%
  dplyr::filter(!(double_entry == 1 & type == "diff")) %>%
  dplyr::select(!double_entry)


# Identify number of items in each category
category_words <- unique(c(pairs$word1, pairs$word2))

#####################################################
####### MEASURE CO-OCCURRENCE ACROSS CORPORA ########
#####################################################

# This section measures co-occurrences between words
# Co-occurrences can range from adjacent to up to 10 words
# apart
# Thus, co-occurrences are measured within an 11-word "window"

# Set the window size
calc_co <- function(window_size) {
  
  # Divide text into ngrams of window_size. Each row is a 
  # window. There is one column for the first word in a window
  # which is called "word1", and another column for the remaining
  # words which is called "context" 
  tidy_skipgrams <- childes_text %>%
    dplyr::ungroup() %>%
    dplyr::select(id, Text) %>%
    tidytext::unnest_tokens(ngram, Text, token = "ngrams", n = window_size) %>%
    dplyr::mutate(ngramID = row_number()) %>%
    tidyr::unite(windowID, ngramID) %>% 
    tidytext::unnest_tokens(word, ngram)
  
  
  # Counts the number of times each word appears with each other word
  # in the same sliding window ngram
  skipgram_counts <- tidy_skipgrams %>%
    # count the number of times each word occurs with each other word
    # in the same sliding window ngram (i.e., co-occur)
    # diag=FALSE to not count words co-occurring with themselves
    widyr::pairwise_count(word, windowID, diag = FALSE, sort = TRUE)
  
  
  # Convert the co-occurrence counts into a co-occurrence matrix
  direct_matrix <- dsm(target = skipgram_counts$item1, feature = skipgram_counts$item2, score = skipgram_counts$n,
                       raw.freq=TRUE, sort=TRUE)
  
  
  
  # Calculate PPMI
  direct_matrix_ppmi <- dsm.score(direct_matrix, score = "MI",
                                  sparse=TRUE, # Convert negative to 0
                                  normalize=FALSE
  )
  
  # Extract ppmi (these are contained in a structure called "S")
  co <- as.data.frame(summary(direct_matrix_ppmi$S))
  co$word1 <- rownames(direct_matrix_ppmi$S)[co$i]
  co$word2 <- colnames(direct_matrix_ppmi$S)[co$j]
  
  # Add a column that identifies the score and window size
  ppmi_name <- paste("ppmi", "_", window_size, sep = "")
  
  colnames(co)[colnames(co) == "x"] <- ppmi_name
  
  
  # This is the table of co-occurrence regularities between word pairs
  co <- co[,c("word1", "word2", ppmi_name)]
  
  # Remove co-occurrence of words with themselves
  co <- co %>%
    dplyr::filter(word1 != word2)
  
  
  
  # Filter down to only words with mcdi values
  co_filter <- co[co$word1 %in% mcdi_aoa$word & co$word2 %in% mcdi_aoa$word,]
  
  co_filter <- pivot_longer(co_filter, cols = starts_with("ppmi"), names_to = "measure", values_to = "co_score")
  
  co_filter$window = window_size
  
  return(co_filter)
}

co_filter <- ldply(c(5, 7, 11), calc_co)


###############################################
### GET CONTEXT OVERLAP FROM CO-OCCURRENCE ####
###############################################


# This section defines a function that calculates "context overlap"
# between words. Context overlap is the degree to which any two words
# tend to co-occur with similar sets of other words. 
# The function takes one argument, cutoff, which is the size of the set of 
# co-occurring words. 

calc_overlap <- function(input_co, cutoff = 100) {
  
  
  # For each word, order its co-occurring words in order of co-occurrence
  co_ordered <- input_co %>%
    dplyr::arrange(word1, desc(co_score))
  
  # For word1, get just its top co-occurring set of words
  co_cutoff <- co_ordered %>%
    group_by(word1) %>%
    dplyr::summarise(word2 = list(word2[1:cutoff]))
  
  co_cutoff_list <- co_cutoff$word2
  
  names(co_cutoff_list) = co_cutoff$word1
  
  # For each pair of words, get the number of overlaps in their top co-occurring word lists
  result = crossprod(table(stack(co_cutoff_list)))
  co_overlap <- data.frame(word1 = colnames(result)[col(result)],
                           word2 = rownames(result)[row(result)], 
                           overlap = c(result))
  co_overlap <- co_overlap[co_overlap$word1 != co_overlap$word2,]
  
  # Calculate jaccard index: the total number of overlapping words, 
  # divided by the union of words across the two words' lists
  
  co_overlap$union <- cutoff*2 - co_overlap$overlap
  
  co_overlap$jaccard <- co_overlap$overlap / co_overlap$union
  
  # Get just the jaccard index value
  co_overlap <- co_overlap %>%
    dplyr::select(!c(overlap, union)) %>%
    dplyr::arrange(desc(jaccard))
  
  return(co_overlap)
}



# Define cutoff values
cutoff_values <- c(25, 50, 100)

# Calculate context overlap for each cutoff value
# This returns a dataframe containing context overlap between pairs
# of words
overlap <- co_filter %>%
  group_by(window) %>%
  group_modify(~ {
    map_dfr(cutoff_values, function(cutoff_val) {
      out <- calc_overlap(.x, cutoff = cutoff_val)
      out$cutoff <- cutoff_val
      out
    })
  }) %>%
  ungroup()

# Combine context overlap with the pairs dataframe generated above, 
# which indicates whether pairs of words belong to the same vs 
# different categories
overlap <- inner_join(overlap, pairs)


################################################
#### CONTEXT SIM BTWN SAME VS DIFF CATEGORY ####
################################################

# Now we have calculated context overlap and representational similarity.
# These measures both capture the degree to which pairs of words tend
# to occur in similar contexts.
# In this section, we will use these measures to calculate context leverage:
# i.e., the degree to which a word occurs in similar contexts to words
# similar in meaning that are learned earlier. We will fist calculate
# context leverage from context overlap, then from representational similarity

# Context Leverage from Context Overlap

# Combine dataframe containing context overlap between word pairs
# with AoA (age of acquisiton) for each word in each pair
overlap_aoa <- overlap %>%
  left_join(mcdi_aoa[,c("word", "aoa_fit")], by = c("word1" = "word")) %>%
  dplyr::rename(aoa1 = aoa_fit) %>%
  left_join(mcdi_aoa[,c("word", "aoa_fit")], by = c("word2" = "word")) %>%
  dplyr::rename(aoa2 = aoa_fit)

# Format the same vs diff category column as a factor
overlap_aoa$type <- factor(overlap_aoa$type, levels = c("same", "diff"))

# Organize the columns
overlap_aoa <- overlap_aoa %>%
  dplyr::select(window, cutoff, word1, word2, aoa1, aoa2, category1, category2, type, category, jaccard)

# Calculate context leverage as a variable called "within_overlap"
# This is calculated as a word's overlap with earlier-learned words that are similar 
# in meaning (i.e., same category) minus overlap with earlier-learned words that
# are different in meaning (i.e., diff category)
# In addition, also calculate a control variable called "prop_same" which is
# the proportion of earlier-learned words that are similar in meaning (same category) 
# out of all earlier-learned words
overlap_within <- overlap_aoa %>%
  dplyr::filter(!is.na(aoa1) & !is.na(aoa2)) %>%
  group_by(window, cutoff, word1, category1, aoa1) %>%
  dplyr::summarise(within_overlap = mean(jaccard[type == "same" & aoa2 < aoa1], na.rm = T) - mean(jaccard[type == "diff" & aoa2 < aoa1], na.rm = T),
                   prop_same = log(length(type[type == "same" & aoa2 < aoa1]) / length(type[aoa2 < aoa1 ]) + .0001)  ) %>%
  dplyr::rename(word = word1,
                aoa = aoa1,
                category = category1) %>%
  left_join(control_stats)

###############################################
####### VISUALIZE RELATIONSHIP WITH AOA #######
###############################################

# Graphs of relationship between context leverage and AoA
# The first graph shows context leverage calculated from context
# overlap, and the second shows it calculated from representational
# similarity

overlap_within$intervening = overlap_within$window - 1

facet_labeller <- labeller(
  intervening = function(x) paste("intervening words =", x),
  cutoff = function(x) paste("list size =", x)
)

overlap_lm_param <- ggplot(overlap_within[overlap_within$aoa < 50,], aes(x = within_overlap, y = aoa)) +
  facet_grid(intervening ~ cutoff, labeller = facet_labeller) +
  geom_point(alpha = .4) +
  stat_smooth(method = "lm", color = rgb(0, .2, .9)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  scale_x_continuous(name = "Context Leverage: Context Overlap") +
  scale_y_continuous(name = "Age of Acquisition (months)") +
  stat_cor(p.digits = 3, label.y = 37) +
  theme_bw() + 
  coord_cartesian(ylim = c(16, 38)) +
  theme(strip.background = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_text(size = 12))



pdf(file = "figures/Context Bootstrap & AoA - Parameters.pdf",
    width = 8, height = 8)
overlap_lm_param 
dev.off()