# load packages
library(emmeans)
library(plyr)
library(dplyr)
library(lme4)
library(gridExtra)
library(grid)
library(lmerTest)
library(car)

##### loading #####
# load data, prepare data
# working directory
setwd("//nasac-faculty.isis.unige.ch/MEDECINE_HOME_PAPAT/Neufo/slaats/MPI/Project 3 - Behavioural studies")

# load data 
data <- read.csv('Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp_preprocessed-outliers-REVIEWS.csv', header = TRUE)
stiminfo <- read.csv('Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv', header = TRUE)

# set factors
data$agreement <- factor(data$agreement, levels=c('correct','incorrect'))
data$surprisal <- factor(data$surprisal, levels=c('low','high'))

##### residuals #####
# we fit a model to extract the residuals for word length, word frequency, and sentence order 
# step 1: model comparison for random effects structure
residual_random_effects <- list('(1 + scale(log(word_length)) * scale(word_frequency) + scale(log(sentence_order)) | UserId)',
                                '(1 + scale(log(word_length)) + scale(word_frequency) + scale(log(sentence_order)) | UserId)',
                                '(1 + scale(log(word_length)) + scale(word_frequency) | UserId)',
                                '(1 + scale(log(word_length)) + scale(log(sentence_order))| UserId)',
                                '(1 + scale(word_frequency)  + scale(log(sentence_order))| UserId)',
                                '(1 + scale(word_frequency) | UserId)',
                                '(1 + scale(log(word_length)) | UserId)',
                                '(1 + scale(log(sentence_order)) | UserId)',
                                '(1 | UserId)')


nonsingular <- list()
for (random_effects in residual_random_effects) {
  fmla = paste('logRT ~ scale(log(word_length)) * scale(word_frequency) + scale(log(sentence_order)) + ', random_effects)
  print(fmla)
  
  model <- lmer(eval(parse(text=fmla)), data, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  if (isSingular(model) == FALSE) {
    nonsingular <- append(nonsingular, model)
  }
}

# two have the same number of random effects
# compare using AIC
# best: number 1 ((1 + scale(log(word_length)) + scale(log(sentence_order))| UserId))
AIC(nonsingular[[1]], nonsingular[[2]])

# step 2: model comparison for fixed effects structure
residual_fixed_effects <- list('logRT ~ scale(log(word_length)) * scale(word_frequency) + scale(log(sentence_order)) + ',
                               'logRT ~ scale(log(word_length)) + scale(word_frequency) + scale(log(sentence_order)) +')

fixed <- list()
for (fixed_effects in residual_fixed_effects) {
  fmla = paste(fixed_effects, '(1 + scale(log(word_length)) + scale(log(sentence_order))| UserId)')
  model <- lmer(eval(parse(text=fmla)), data, control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  fixed <- append(fixed, model)
}

anova(fixed[[1]], fixed[[2]])
# interaction is better

summary(fixed[[1]])

# add residuals to the df
data$logRTresidual <- residuals(fixed[[1]])
write.csv(data, 'Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp_residuals_manual-outliers-REVIEWS.csv')
