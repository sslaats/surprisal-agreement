# load packages
library(emmeans)
library(plyr)
library(dplyr)
library(lme4)
library(gridExtra)
library(grid)
library(lmerTest)
library(car)
library(MAP)
library(broom)
library(tidyverse)

emm_options(lmerTest.limit = 3470)

##### loading #####
# load data, prepare data
# working directory
setwd("K:/Project/Project 3 - Behavioural studies")

############################################ WORD INDEX ANALYSIS ############################################
# load data 
data <- read.csv('K:/Project/Project 3 - Behavioural studies/3 - SPR main/analysis/readingtimes_88pp_residuals_manual.csv', header = TRUE)
stiminfo <- read.csv('K:/Project/Project 3 - Behavioural studies/5 - metadata/frequency_length_position.csv', header = TRUE)

# set factors
data$agreement <- factor(data$agreement, levels=c('correct','incorrect'))
data$surprisal <- factor(data$surprisal, levels=c('low','high'))
data$correct_number <- factor(data$correct_number, levels=c("singular", "plural"))

##### all effects structures #####
random_effects <- list('(1 + agreement * surprisal + agreement * correct_number | UserId)',
                       '(1 + surprisal + agreement * correct_number | UserId)',
                       '(1 + agreement * correct_number | UserId)',
                       '(1 + agreement * surprisal + correct_number | UserId)',
                       '(1 + agreement * surprisal | UserId)',
                       '(1 + surprisal + agreement + correct_number | UserId)',
                       '(1 + agreement + correct_number | UserId)',
                       '(1 + surprisal + correct_number | UserId)',
                       '(1 + surprisal + agreement | UserId)',
                       '(1 + surprisal | UserId)',
                       '(1 + agreement | UserId)',
                       '(1 + correct_number | UserId)',
                       '(1 | UserId)')

con_random_effects <- list('(1 + agreement * surprisal_value + agreement * correct_number | UserId)',
                           '(1 + surprisal_value + agreement * correct_number | UserId)',
                           '(1 + agreement * correct_number | UserId)',
                           '(1 + agreement * surprisal_value + correct_number | UserId)',
                           '(1 + agreement * surprisal_value | UserId)',
                           '(1 + surprisal_value + agreement + correct_number | UserId)',
                           '(1 + agreement + correct_number | UserId)',
                           '(1 + surprisal_value + correct_number | UserId)',
                           '(1 + surprisal_value + agreement | UserId)',
                           '(1 + surprisal_value | UserId)',
                           '(1 + agreement | UserId)',
                           '(1 + correct_number | UserId)',
                           '(1 | UserId)')

fixed_effects <- list('logRTresidual ~ agreement * surprisal + agreement * correct_number +', # 1
                      'logRTresidual ~ surprisal + agreement * correct_number +',             # 2
                      'logRTresidual ~ agreement * correct_number +',                         # 3
                      'logRTresidual ~ agreement * surprisal + correct_number +',             # 4
                      'logRTresidual ~ agreement * surprisal +',                              # 5
                      'logRTresidual ~ surprisal + agreement + correct_number +',             # 6
                      'logRTresidual ~ agreement + correct_number +',                         # 7
                      'logRTresidual ~ surprisal + correct_number +',                         # 8
                      'logRTresidual ~ surprisal + agreement +',                              # 9
                      'logRTresidual ~ surprisal +',                                          # 10
                      'logRTresidual ~ agreement +',                                          # 11
                      'logRTresidual ~ correct_number +',                                     # 12
                      'logRTresidual ~ ')                                                     # 13

con_fixed_effects <- list('logRTresidual ~ agreement * surprisal_value + agreement * correct_number +',
                          'logRTresidual ~ surprisal_value + agreement * correct_number +',
                          'logRTresidual ~ agreement * correct_number +',
                          'logRTresidual ~ agreement * surprisal_value + correct_number +',
                          'logRTresidual ~ agreement * surprisal_value +',
                          'logRTresidual ~ surprisal_value + agreement + correct_number +',
                          'logRTresidual ~ agreement + correct_number +',
                          'logRTresidual ~ surprisal_value + correct_number +',
                          'logRTresidual ~ surprisal_value + agreement +',
                          'logRTresidual ~ surprisal_value +',
                          'logRTresidual ~ agreement +',
                          'logRTresidual ~ correct_number +',
                          'logRTresidual ~ ')

##### models with categorical surprisal #####

##### target word #####

# step 1: reduce random effects
nonsingular5 <- list()
for (random_effect in random_effects) {
  
  fmla = eval(parse(text=paste(fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index== 5), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    nonsingular5 <- append(nonsingular5, model)
  }
}

# best model: "(1 + agreement | UserId)"

# step 2: reduce fixed effects
fixed5 <- list()
for (fixed_effect in fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 + agreement | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 5), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  fixed5 <- append(fixed5, model)
}

anova(fixed5[[1]], fixed5[[2]]) # not significant, interaction between agreement & surprisal can go
anova(fixed5[[2]], fixed5[[3]]) # not significant, surprisal can go
anova(fixed5[[3]], fixed5[[7]]) # model 3 is better than model 7 - interaction agreement & plurality should stay
                                # we stop here

# interaction between plurality and agreement
summary(fixed5[[3]])

# effect of agreement significant for both, but in the wrong direction for 
# when plural is the correct form

# agreement by number
emmeans(fixed5[[3]], specs='agreement', by='correct_number', pbkrtest.limit = 3470)
joint_tests(fixed5[[3]], by='correct_number', adjust='none', pbkrtest.limit = 3470)

# number by agreement
emmeans(fixed5[[3]], specs='correct_number', by='agreement',pbkrtest.limit = 3470)
joint_tests(fixed5[[3]], by='agreement', adjust='none', pbkrtest.limit = 3470)

################################################################################
##### spillover - 1 #####

# step 1: reduce random effects
nonsingular6 <- list()
for (random_effect in random_effects) {
  
  fmla = eval(parse(text=paste(fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index== 6), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    nonsingular6 <- append(nonsingular6, model)
  }
}

# best model: "(1 + agreement | UserId)"

# step 2: reduce fixed effects
fixed6 <- list()
for (fixed_effect in fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 + agreement | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 6), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  fixed6 <- append(fixed6, model)
}

anova(fixed6[[1]], fixed6[[2]]) # no difference, interaction surprisal & agreement can go
anova(fixed6[[2]], fixed6[[3]]) # no difference, surprisal can go
anova(fixed6[[3]], fixed6[[7]]) # interaction between agreement and number should stay
                                # we stop here

# model with agreement is best
summary(fixed6[[3]])

# agreement by number
emmeans(fixed6[[3]], specs='agreement', by='correct_number')
joint_tests(fixed6[[3]], by='correct_number', pbkrtest.limit=3470, adjust='none')

# effect of agreement exists for both

# number by agreement
emmeans(fixed6[[3]], specs='correct_number', by='agreement')
joint_tests(fixed6[[3]], by='agreement', pbkrtest.limit=3470, adjust='none')

# effect of number exists only in correct 

##### spillover - 2 #####
nonsingular7 <- list()
for (random_effect in random_effects) {
  
  fmla = eval(parse(text=paste(fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index== 7), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    nonsingular7 <- append(nonsingular7, model)
  }
}

# best model: random intercept "(1 | UserId)"

# step 2: reduce fixed effects
fixed7 <- list()
for (fixed_effect in fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 7), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  fixed7 <- append(fixed7, model)
}

anova(fixed7[[1]], fixed7[[2]]) # no difference, interaction between surprisal & agremeent can go
anova(fixed7[[2]], fixed7[[3]]) # main effects is better, surprisal should stay
anova(fixed7[[2]], fixed7[[6]]) # interaction between agreement and number can go
anova(fixed7[[6]], fixed7[[9]]) # correct_number should stay
anova(fixed7[[6]], fixed7[[8]]) # agreement should stay

# model with main effects is best
summary(fixed7[[6]])
# main effects of agreement & surprisal, both lead to longer RTs

##### spillover - 3 #####
nonsingular8 <- list()
for (random_effect in random_effects) {
  
  fmla = eval(parse(text=paste(fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index== 8), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    nonsingular8 <- append(nonsingular8, model)
  }
}

# best model: "(1 + | UserId)"

# step 2: reduce fixed effects
fixed8 <- list()
for (fixed_effect in fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 8), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  fixed8 <- append(fixed8, model)
}

anova(fixed8[[1]], fixed8[[2]]) # no difference, interaction between agreement & surprisal can go
anova(fixed8[[2]], fixed8[[3]]) # no difference, surprisal can go
anova(fixed8[[3]], fixed8[[7]]) # interaction agreement & correct number can go
anova(fixed8[[7]], fixed8[[11]]) # correct_number can go
anova(fixed8[[11]], fixed8[[13]]) # agreement should stay

summary(fixed8[[11]]) # main effect of agreement

##### models with continuous surprisal #####
##### target word #####
# step 1: reduce random effects
con_nonsingular5 <- list()
for (random_effect in con_random_effects) {
  
  fmla = eval(parse(text=paste(con_fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index== 5), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    con_nonsingular5 <- append(con_nonsingular5, model)
  }
}

# best model: "(1 + agreement | UserId)"

# step 2: reduce fixed effects
con_fixed5 <- list()
for (fixed_effect in con_fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 + agreement | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 5), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  con_fixed5 <- append(con_fixed5, model)
}

anova(con_fixed5[[1]], con_fixed5[[2]]) # no difference, interaction can go
anova(con_fixed5[[2]], con_fixed5[[3]]) # main is better, surprisal should stay
anova(con_fixed5[[2]], con_fixed5[[6]]) # interaction agreement & number should stay

# model with surprisal and interaction between agreement and number
summary(con_fixed5[[2]])

# agreement by number
emmeans(con_fixed5[[2]], specs='agreement', by='correct_number', pbkrtest.limit = 3470)
joint_tests(con_fixed5[[2]], by='correct_number', pbkrtest.limit = 3470, adjust='none')

# number by agreement
emmeans(con_fixed5[[2]], specs='correct_number', by='agreement', pbkrtest.limit = 3470)
joint_tests(con_fixed5[[2]], by='agreement', pbkrtest.limit = 3470, adjust='none')

##### spillover - 1 #####
# step 1: reduce random effects
con_nonsingular6 <- list()
for (random_effect in con_random_effects) {
  
  fmla = eval(parse(text=paste(con_fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index == 6), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    con_nonsingular6 <- append(con_nonsingular6, model)
  }
}

# both surprisal and agreement in random effects structure are not singular
# AIC reveals that random effect for agreement is better
AIC(con_nonsingular6[[1]], con_nonsingular6[[2]])

# step 2: reduce fixed effects
con_fixed6 <- list()
for (fixed_effect in con_fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 + agreement | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 6), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  con_fixed6 <- append(con_fixed6, model)
}

anova(con_fixed6[[1]], con_fixed6[[2]]) # marginal, p = 0.065

# branch 1: reduce from interaction with surprisal and agreement
anova(con_fixed6[[1]], con_fixed6[[4]]) # the interaction between number and agreement should stay
summary(con_fixed6[[1]])

# agreement by number
emmeans(con_fixed6[[1]], specs='agreement', by='correct_number',pbkrtest.limit = 3470)
joint_tests(con_fixed6[[1]], by='correct_number', pbkrtest.limit = 3470, adjust='none')

# number by agreement
emmeans(con_fixed6[[1]], specs='correct_number', by='agreement',pbkrtest.limit = 3470)
joint_tests(con_fixed6[[1]], by='agreement',pbkrtest.limit = 3470, adjust='none')

# surprisal by agreement
emmeans(con_fixed6[[1]], specs='surprisal_value', by='agreement',pbkrtest.limit = 3470)
joint_tests(con_fixed6[[1]], by='agreement', pbkrtest.limit = 3470, adjust='none')

# agreement by surprisal


# branch 2: reduce from main effects for surprisal and agreement
anova(con_fixed6[[2]], con_fixed6[[3]]) # main is better, surprisal should stay
anova(con_fixed6[[2]], con_fixed6[[6]]) # interaction between agreement & number should stay

# main effects model wins
summary(con_fixed6[[2]])



##### spillover - 2 #####
# step 1: reduce random effects
con_nonsingular7 <- list()
for (random_effect in con_random_effects) {
  
  fmla = eval(parse(text=paste(con_fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index == 7), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    con_nonsingular7 <- append(con_nonsingular7, model)
  }
}

# random slopes for correct_number
# this leads to singular fit in many of the other models, so random intercept it is

# step 2: reduce fixed effects
con_fixed7 <- list()
for (fixed_effect in con_fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 | UserId)")))
  model <- lmer(fmla, subset(data, word_index== 7), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  con_fixed7 <- append(con_fixed7, model)
}

anova(con_fixed7[[1]], con_fixed7[[2]]) # no difference, interaction can go
anova(con_fixed7[[2]], con_fixed7[[3]]) # main is better, surprisal should stay
anova(con_fixed7[[2]], con_fixed7[[6]]) # interaction between agreement & number can go
anova(con_fixed7[[6]], con_fixed7[[9]]) # correct_number should stay
anova(con_fixed7[[6]], con_fixed7[[8]]) # agreement should stay

# main effects model wins
summary(con_fixed7[[6]])

##### spillover - 3 #####
# singular fit
# step 1: reduce random effects
con_nonsingular8 <- list()
for (random_effect in con_random_effects) {
  
  fmla = eval(parse(text=paste(con_fixed_effects[[1]], random_effect)))
  model <- lmer(fmla, subset(data, word_index == 8), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  
  if (isSingular(model) == FALSE) {
    print(random_effect)
    con_nonsingular8 <- append(con_nonsingular8, model)
  }
}

# only a random intercept model is non-singular

# step 2: reduce fixed effects
con_fixed8 <- list()
for (fixed_effect in con_fixed_effects) {
  fmla = eval(parse(text=paste(fixed_effect, "(1 | UserId)")))
  model <- lmer(fmla, subset(data, word_index == 8), control=lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5)), REML=FALSE)
  con_fixed8 <- append(con_fixed8, model)
}

anova(con_fixed8[[1]], con_fixed8[[2]]) # no difference, interaction can go
anova(con_fixed8[[2]], con_fixed8[[3]]) # no difference, surprisal can go
anova(con_fixed8[[3]], con_fixed8[[7]]) # no difference, interaction agreement & number can go
anova(con_fixed8[[7]], con_fixed8[[11]]) # no difference, number can go
anova(con_fixed8[[11]], con_fixed8[[13]]) # agreement should stay

# agreement only wins
summary(con_fixed8[[11]])
