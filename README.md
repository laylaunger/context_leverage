# Context-Leverage

This repository includes scripts and data for the manuscript, Do Young Children Learn Words from the Company they Keep?

This project examines how children learn new words just from encountering them in language. For example, a child might get the sense that a mango is a fruit just from hearing it in the context of words that tend to occur with other fruit words, such as "sweet" and "juicy". We refer to this route to word learning as "context leverage".

The goal of this project is to examine whether context leverage is an important route for real-world learning. The project tackles this goal using the following logic. First, children's everyday language experiences might provide stronger context leverage support for learning some words than others. For example, "mango" might have a strong tendency to occur in similar contexts to other fruit words, whereas "melon" might have only a weak tendency to occur in similar contexts to other fruit words. If context leverage is an important route for word learning, then words with strong context leverage support in everyday language should be learned earlier in childhood than words with weaker support.

The focus of this project is to measure context leverage support in children's everyday language input, and text whether it predicts word learning.

All scripts for this analysis are in the scripts subfolder and have been extensively commented for reproducibility.

The main script for this analysis is:

context_leverage.R

This script can be run without running any of the others. The additional scripts include supplemental analyses, and scripts for generating one of the measures used in the context_leverage.R script by training RNNs on children's langauge input.
