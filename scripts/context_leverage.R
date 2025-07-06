#####################################################
#################### DESCRIPTION ####################
#####################################################

# Measures of context leverage capture the degree to which words
# occur in smilar contexts to words similar in meaning that are
# learned earlier in development. 

# The focus of this script is to generate multiple measures 
# of context leverage in corpora of language input to children,
# and test whether it predicts early word learning above
# and beyond other predictors such as frequency.

# Prior to running this script, you should ensure that you
# have the data folder, which includes: 
# (1) a file containing utterance statistics: utt_stats_english.rds
# (2) a file containing age of acquisition data: aoa_mcdi_english.csv
# (3) a file containing a list of common first names that 
#     will be replaced with a common token: childes_names.txt
# (4) a file with concreteness ratings: abcon_mcdi_english.csv
# (5) a file linking words to semantic categories: categories_english.csv
# (6) a file containing names of RNN models that were trained on the CHILDES:
#     corpus: rnn_model_ids.txt


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
library(ggrepel)
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
################## GET MODELS ###################
#####################################################

# Get urls for files that contain pairwise similarities between representations of 
# words formd by training RNNs on the CHILDES corpus
model_ids <- readLines("data/rnn_model_ids.txt")

model_urls <- paste0("https://rnn-childes-similarities.s3.us-east-2.amazonaws.com/", model_ids)

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
window_size = 11

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

co_filter <- pivot_longer(co_filter, cols = "ppmi_11", names_to = "measure", values_to = "co_score")


###############################################
### GET CONTEXT OVERLAP FROM CO-OCCURRENCE ####
###############################################


# This section defines a function that calculates "context overlap"
# between words. Context overlap is the degree to which any two words
# tend to co-occur with similar sets of other words. 
# The function takes one argument, cutoff, which is the size of the set of 
# co-occurring words. 

calc_overlap <- function(cutoff = 100) {
  
  
  # For each word, order its co-occurring words in order of co-occurrence
  co_ordered <- co_filter %>%
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

# Calculate context overlap for the default cutoff value of 100
# This returns a dataframe containing context overlap between pairs
# of words
overlap <- calc_overlap()

# Combine context overlap with the pairs dataframe generated above, 
# which indicates whether pairs of words belong to the same vs 
# different categories
overlap <- inner_join(overlap, pairs)

###############################################
####### GET REPRESENTATIONAL SIMILARITY #######
###############################################

# This section calculates "representational similarity" between words.
# Representational similarity is calculated from the representations of
# words in recurrent neural network models (RNNs) that have been trained
# on the corpus of child language input. 
# Because training these models takes a decent amount of time (for this
# work, training took ~ 5 hours using supercomputing resources), models
# have been trained and the representation of each word in each model
# has been extracted. 
# There are representations from 4 models that have the same architecture
# and training regime, but were each randomly initialized. The output
# from each model is identified with a randomly generated string



# Calculate cosine similarity between word representations in each model
get_sim <- function(input_file){
  
  # Read file
  input_matrix <- readRDS(url(input_file))
  
  # Convert to pairwise similarities in dataframe
  similarities <- data.frame(word1 = colnames(input_matrix)[col(input_matrix)],
                             word2 = rownames(input_matrix)[row(input_matrix)], 
                             similarity = c(input_matrix))
  
  # Remove similarities between words and themselves
  similarities <- similarities %>%
    dplyr::filter(word1 != word2)
  
  # Combine similarities between words with the pairs dataframe,
  # which identifies whether word pairs are from the same vs different
  # category
  similarities <- inner_join(similarities, pairs)
  
  # Get the model name
  similarities$model <- str_extract(input_file, pattern = "id.{6}")
  
  similarities$model_type <- "rnn"
  
  return(similarities)
  
}

# Get similarities from each model
rep_sim <- ldply(model_urls, get_sim)

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
  dplyr::select(word1, word2, aoa1, aoa2, category1, category2, type, category, jaccard)

# Calculate context leverage as a variable called "within_overlap"
# This is calculated as a word's overlap with earlier-learned words that are similar 
# in meaning (i.e., same category) minus overlap with earlier-learned words that
# are different in meaning (i.e., diff category)
# In addition, also calculate a control variable called "prop_same" which is
# the proportion of earlier-learned words that are similar in meaning (same category) 
# out of all earlier-learned words
overlap_within <- overlap_aoa %>%
  dplyr::filter(!is.na(aoa1) & !is.na(aoa2)) %>%
  group_by(word1, category1, aoa1) %>%
  dplyr::summarise(within_overlap = mean(jaccard[type == "same" & aoa2 < aoa1], na.rm = T) - mean(jaccard[type == "diff" & aoa2 < aoa1], na.rm = T),
                   prop_same = log(length(type[type == "same" & aoa2 < aoa1]) / length(type[aoa2 < aoa1 ]) + .0001)  ) %>%
  dplyr::rename(word = word1,
                aoa = aoa1,
                category = category1) %>%
  left_join(control_stats)


# Context Leverage from Representational SImilarity

# Combine dataframe containing representational similarity between word pairs
# with AoA (age of acquisiton) for each word in each pair
rep_aoa <- rep_sim %>%
  left_join(mcdi_aoa[,c("word", "aoa_fit")], by = c("word1" = "word")) %>%
  dplyr::rename(aoa1 = aoa_fit) %>%
  left_join(mcdi_aoa[,c("word", "aoa_fit")], by = c("word2" = "word")) %>%
  dplyr::rename(aoa2 = aoa_fit)

# Format the same vs diff category column as a factor
rep_aoa$type <- factor(rep_aoa$type, levels = c("same", "diff"))

# Organize the columns
rep_aoa <- rep_aoa %>%
  dplyr::select(word1, word2, aoa1, aoa2, category1, category2, type, category, model, similarity)

# Calculate context leverage as a variable called "within_rep"
# This is calculated as a word's representational similarity with earlier-learned 
# words that are similar  in meaning (i.e., same category) minus similarity with 
# earlier-learned words that are different in meaning (i.e., diff category)
# In addition, also calculate a control variable called "prop_same" which is
# the proportion of earlier-learned words that are similar in meaning (same category) 
# out of all earlier-learned words
rep_within <- rep_aoa %>%
  dplyr::filter(!is.na(aoa1) & !is.na(aoa2)) %>%
  group_by(word1, category1, aoa1, model) %>%
  dplyr::summarise(within_rep = mean(similarity[type == "same" & aoa2 < aoa1], na.rm = T) - mean(similarity[type == "diff" & aoa2 < aoa1], na.rm = T),
                   prop_same = log(length(type[type == "same" & aoa2 < aoa1]) / length(type[aoa2 < aoa1 ]) + .0001)  ) %>%
  dplyr::rename(word = word1,
                aoa = aoa1,
                category = category1) %>%
  left_join(control_stats)


###############################################
#### CORRELATION BETWEEN CONTEXT VARIABLES ####
###############################################

# This section looks at the correlation between context leverage derived from 
# context overlap and representational similarity

# Combine the two metrics
var_within <- rep_within %>%
  ungroup() %>%
  dplyr::select(word, model, within_rep) %>%
  left_join(overlap_within[,c("word", "within_overlap")])


# Calculate correlations between the two metrics
within_correlations <- var_within %>%
  group_by(model) %>%
  dplyr::summarize(cor_estimate=cor.test(within_rep, within_overlap)$estimate,
                   cor_p=round(cor.test(within_rep, within_overlap)$p.value, 2))


# Plot the relationship between them
corr_plot <- ggplot(var_within, aes(x = within_overlap, y = within_rep)) +
  geom_point(aes(shape = model), alpha = .4, show.legend = F) +
  stat_smooth(aes(group = model), method = "lm", color = rgb(0, .2, .9), alpha = .2) +
  scale_x_continuous(name = "Context Overlap") +
  scale_y_continuous(name = "Representational Similarity") +
  theme_bw() +
  theme(strip.background = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_text(size = 12))

pdf(file = "figures/Context Bootstrap Correlation.pdf",
    width = 4, height = 3.25)
corr_plot
dev.off()


###############################################
###### CORRELATION BETWEEN ALL VARIABLES ######
###############################################

# This section looks at the correlation between the context leverage and
# control variables

# Combine the two metrics
var_all <- rep_within %>%
  ungroup() %>%
  dplyr::select(word, model, within_rep) %>%
  dplyr::filter(model == unique(model)[2]) %>%
  left_join(overlap_within)

# Calculate correlation matrix
corr_mat <- cor(var_all[, c("utt_length", "abcon", "log_freq", "prop_same", "within_overlap", "within_rep")],
               use = "complete.obs")

# Set custom names
custom_names <- c("Utterance Length", "Concreteness", "Log Frequency", "Meaning Leverage", "Context Overlap", "Representational Similarity")

# Assign as row and column names
colnames(corr_mat) <- custom_names
rownames(corr_mat) <- custom_names

corr_mat_plot <- ggcorrplot(cor_mat,
           type = "upper",
           lab = T,
           outline.col = "transparent") +
  scale_x_discrete(name = "") +
  scale_y_discrete(name = "") +
  theme_bw() +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        legend.title = element_blank(),
        legend.key.height = unit(0.08, 'npc'),
        axis.text = element_text(size = 10, color = "black"))

pdf(file = "figures/Context Bootstrap Corr Matrix.pdf",
    width = 6, height = 6)
corr_mat_plot
dev.off()


###############################################
####### VISUALIZE RELATIONSHIP WITH AOA #######
###############################################

# Graphs of relationship between context leverage and AoA
# The first graph shows context leverage calculated from context
# overlap, and the second shows it calculated from representational
# similarity

# Add a column indicating a subset of words to use as labeled examples in the plot
overlap_within$label <- ifelse(overlap_within$word %in% c("bucket", "push", "toe", "cereal"),
                           overlap_within$word, '')


# Add a column indicating a subset of words to use as labeled examples in the plot
rep_within$label <- ifelse(rep_within$word %in% c("dark", "goose", "strawberry", "chair"),
                           rep_within$word, '')


overlap_lm <- ggplot(overlap_within[overlap_within$aoa < 50,], aes(x = within_overlap, y = aoa)) +
  geom_point(alpha = .4) +
  stat_smooth(method = "lm", color = rgb(0, .2, .9)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  geom_label_repel(aes(label = label), fill = "#f2919b",
                   point.padding = .1, max.overlaps = 500, force = 10, min.segment.length = 0,
                   force_pull = .0001,
                   show.legend = F) +
  scale_x_continuous(name = "Context Leverage: Context Overlap") +
  scale_y_continuous(name = "Age of Acquisition (months)") +
  theme_bw() + 
  coord_cartesian(ylim = c(16, 35)) +
  theme(strip.background = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_text(size = 12))



rep_lm <- ggplot(rep_within[rep_within$aoa < 50 & rep_within$word != "play" & rep_within$word != "hurt",], aes(x = within_rep, y = aoa)) +
  geom_point(aes(shape = model), alpha = .4, show.legend = F) +
  stat_smooth(aes(group = model), method = "lm", color = rgb(0, .2, .9), alpha = .2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  geom_label_repel(data = rep_within[rep_within$model == unique(rep_within$model)[2],],
                     aes(label = label), fill = "#f2919b",
                   point.padding = .1, max.overlaps = 500, force = 10, min.segment.length = 0,
                   force_pull = .0001,
                   show.legend = F) +
  scale_x_continuous(name = "Context Leverage: Representational Similarity") +
  scale_y_continuous(name = "Age of Acquisition (months)") +
  theme_bw() + 
  coord_cartesian(ylim = c(16, 35)) +
  theme(strip.background = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_text(size = 12))


overlap_lm
rep_lm

cowplot::plot_grid(overlap_lm, rep_lm)

pdf(file = "figures/Context Bootstrap & AoA.pdf",
    width = 8, height = 3.25)
cowplot::plot_grid(overlap_lm, rep_lm)
dev.off()


###############################################
############# STATISTICAL ANALYSIS ############
###############################################

# Analyses predicting AoA from context leverage and control variables
# Analyses are conducted separately for context overlap and representational
# similarity


# Scale variables
overlap_within_scale <- overlap_within
overlap_within_scale[,c("log_freq", "log_freq_final", "utt_length", "abcon", "prop_same", "within_overlap")] <- 
  scale(overlap_within_scale[,c("log_freq", "log_freq_final", "utt_length", "abcon", "prop_same", "within_overlap")])


rep_within_scale <- rep_within
rep_within_scale[,c("log_freq", "log_freq_final", "utt_length", "abcon", "prop_same", "within_rep")] <- 
  scale(rep_within_scale[,c("log_freq", "log_freq_final", "utt_length", "abcon", "prop_same", "within_rep")])


############### CONTEXT OVERLAP ################

# Fit model predicting AoA from context overlap leverage and control variables,
# then get coefficients for predictors
overlap_model <- lm(aoa ~ log_freq + utt_length + abcon + prop_same + within_overlap, data = overlap_within_scale)

summary(overlap_model)

# Model coefficients for predictors: Context Overlap
overlap_coef <- tidy(overlap_model, conf.int = T) %>%
  dplyr::filter(term != "(Intercept)")

overlap_coef$term <- factor(overlap_coef$term, levels = c("utt_length", "log_freq",
                                                          "abcon",
                                                          "prop_same", "within_overlap"))


######## REPRESENTATIONAL SIMILARITY ##########

# Fit model predicting AoA from representational similarity leverage and control variables,
# then get coefficients for predictors. This code is slightly different from
# above because it fits models and gets coefficients together within the same
# tidy code which applies these steps to the representational similarity metrics
# derived from each of the 4 randomly initialized models
rep_coef <- rep_within_scale %>%
  group_by(model) %>%  
  do(tidy(lm(aoa~log_freq + utt_length + abcon + prop_same + within_rep, data=.), conf.int = T)) %>%
  dplyr::filter(term != "(Intercept)")


rep_coef$term <- factor(rep_coef$term, levels = c("utt_length", "log_freq",
                                                  "abcon",
                                                  "prop_same", "within_rep"))

# Randomly order the models (this step is just for aesthetics in the
# graph below)
rep_coef$model <- factor(rep_coef$model, levels = sample(unique(rep_coef$model)))


###############################################
############### VISUALIZE FITS ################
###############################################

# Plot the coefficients and confidence intervals for each
# predictor from the analyses above

overlap_outcome <- ggplot(overlap_coef, aes(estimate, term, xmin = conf.low, xmax = conf.high, height = 0)) +
  geom_point(aes(color = term), show.legend = F) +
  geom_errorbarh(aes(color = term), show.legend = F) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_x_continuous(name = expression("Earlier AoA" %<->% "Later AoA")) +
  scale_y_discrete(name ="", labels = c("Mean Utterance Length", "Log Frequency", "Concreteness", "Meaning Leverage", "Context Leverage")) +
  ggtitle("Context Overlap") +
  coord_cartesian(xlim = c(-2, 2)) +
  theme_bw() +
  theme(strip.background = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_text(size = 12),
        axis.text.y = element_text(size = 11),
        axis.text = element_text(color = "black"))



rep_outcome <- ggplot(rep_coef, aes(estimate, term, group = model, xmin = conf.low, xmax = conf.high, height = 0)) +
  geom_point(aes(color = term), show.legend = F, position = position_dodge(width = .2)) +
  geom_errorbarh(aes(color = term), show.legend = F, position = position_dodge(width = .2)) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  scale_x_continuous(name = expression("Earlier AoA" %<->% "Later AoA")) +
  scale_y_discrete(name ="", labels = c("Mean Utterance Length", "Log Frequency", "Concreteness", "Meaning Leverage", "Context Leverage")) +
  coord_cartesian(xlim = c(-2.1, 2.1)) +
  ggtitle("Representational Similarity") + 
  theme_bw() +
  theme(strip.background = element_blank(),
        panel.grid = element_blank(),
        axis.title = element_text(size = 12),
        axis.text.y = element_text(size = 11),
        axis.text = element_text(color = "black"))



plot_grid(overlap_outcome, rep_outcome)

pdf(file = "figures/Context Bootstrap Model Output.pdf",
    height = 8.75, width = 6)
plot_grid(overlap_outcome, rep_outcome, ncol = 1)
dev.off()





