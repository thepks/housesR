library(readr)
library(purrr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(gbm)
library(forcats)
library(modelr)
#library(psych)
library(devtools)
install_github("thepks/sapr")
library(gptools)

#functions
# the evalaution method
localRMSE <- function(pred, obs,model,lev) {sqrt(mean((log(pred) - log(obs))^2))}

# convenience method used to rename  columns that start with numbers
mod_name <- function(f) { if(!is.na(str_match(f,"^[0-9].*"))){paste0("val_",f)} else {f}}

# check columns for na
check_na <- function(f) {sum(is.na(f))}


# function to clean data
clean_data <- function (o) {

  #alter names to start with char
  names(o) <- lapply(names(o),mod_name)

  absent_items <- sap_missing_values(o)

  # Engineer values
  o$age_indicator <- o$YrSold - o$YearBuilt
  o$YearRemodAdd_indicator <- o$YrSold - o$YearRemodAdd
  o$YearBuilt <- NULL
  o$YearRemodAdd <- NULL
  o$garage_age_indicator <- o$YrSold - o$GarageYrBlt
  
  o$GarageYrBlt <- NULL
  #house_raw[house_no_gar,]$garage_age_indicator <- -1
  
  # Set some of the absents to None
  
  o[is.na(o$PoolQC),]$PoolQC <- 'No Pool'
  o[is.na(o$Alley),]$Alley <- 'No Alley Access'
  o[is.na(o$Fence),]$Fence <- 'No Fence'

  house_no_bsmt <- is.na(o$BsmtQual)
  o[house_no_bsmt,]$BsmtQual <- "No Basement"
  o[house_no_bsmt,]$BsmtCond <- "No Basement"
  o[house_no_bsmt,]$BsmtExposure <- "No Basement"
  o[house_no_bsmt,]$BsmtFinType1 <- "No Basement"
  o[house_no_bsmt,]$BsmtFinSF1 <- 0
  o[house_no_bsmt,]$BsmBsmtFinType2 <- "No Basement"
  o[house_no_bsmt,]$BsmtUnfSF <- 0
  o[house_no_bsmt,]$TotalBsmtSF <- 0
  o[is.na(o$BsmtFinType2),]$BsmtFinType2 <- "No 2nd Basement"

  o[is.na(o$FireplaceQu),]$FireplaceQu <- 'No Fireplace'
  
  o[is.na(o$MiscFeature),]$MiscFeature <- 'None'
  
  house_no_electric <- is.na(o$Electrical) | o$Electrical < 1
  if(sum(house_no_electric)>1) { o[is.na(house_no_electric),]$Electrical <- 0}
  
  
  house_no_gar <- is.na(o$GarageType)
  
  o[house_no_gar,]$GarageCond <- "No Garage"
  o[house_no_gar,]$GarageQual <- "No Garage"
  o[house_no_gar,]$GarageFinish <- "No Garage"
  o[house_no_gar,]$GarageType <- "No Garage"
  o[house_no_gar,]$garage_age_indicator <- -1
  o[house_no_gar,]$GarageCars <- 0
  
  o[is.na(o$LotFrontage),]$LotFrontage <- 0
  
  
  iffy <- ifelse(is.na(o$MasVnrType) | o$MasVnrType=="None",TRUE,FALSE)
  o[iffy,]$MasVnrArea <- 0
  

  # sort out the factors
  fields_class <- sapply(o,class)
  fields_class[["MSSubClass"]] <- "character"
  fields_class[["OverallQual"]] <- "character"
  fields_class[["OverallCond"]] <- "character"
  fields_class <- as.data.frame(fields_class, attr(fields_class,"names"))
  fields_class$Id <- attr(fields_class,"row.names")
  char_fields <- fields_class[fields_class$fields_class == "character",]

  absent_items <- sap_missing_values(o)
  absent_char_items <- absent_items[(absent_items$feature) %in% char_fields$Id,]
  absent_num_items <- absent_items[!(absent_items$feature %in% char_fields$id),]

  cleanup <- cbind(absent_char_items$feature,"None")
  absent_num_means <- as.list(colMeans(o[,!(names(o) %in% char_fields$Id)], na.rm=TRUE))
  cleanup2 <- cbind(attr(absent_num_means,"name"),absent_num_means)
  names(cleanup2) <- NULL
  cleanup <- rbind(cleanup, cleanup2)
  row.names(cleanup) <- cleanup[,1]
  
  o <- replace_na(o,as.list(cleanup[,2]))

  
  o[,char_fields$Id] <- lapply(o[,char_fields$Id],factor)
    
  return(o)
  
}

# Read in raw data
house_raw <- read_csv('train.csv')

# number of houses
nrow(house_raw)

#number of attributes
ncol(house_raw)-1

house_raw <- clean_data(house_raw)

# recheck

absent_items <- sap_missing_values(house_raw)
absent_items %>% ggplot() + geom_col(aes(reorder(feature,order(absent_items$gaps,decreasing = TRUE)),gaps,fill=gaps)) + coord_flip() + labs(title="Potential Data Gaps", x="Feature")

absent_items

# find numeric fields
# then this is only good for numerics
class_fields <- lapply(house_raw,class)
factor_filter <- class_fields != "factor"
numeric_fields <- names(house_raw)[factor_filter]

# what about constant columns, all taken care of?
apply(house_raw[,numeric_fields], 2, var, na.rm = TRUE)
sum(apply(house_raw[,numeric_fields], 2, var, na.rm=TRUE) ==0)

# split the data into 3 sets 60:20:20

set.seed(11)
training_data <- resample_partition(house_raw,c(train=0.6, trial=0.4))

# first reduce dimensions using pca
# to do this going to use columns 2 to the SalePrice

pca_data <- as.data.frame(training_data$train)

# then this is only good for numerics
pca_data <- pca_data[,factor_filter]
response_column <- which(names(pca_data) == 'SalePrice')

pca_data <- pca_data[,!(names(pca_data) %in% c("Id","SalePrice"))]


# now in this set constant columns can occur
drop_cols <- !near(apply(pca_data, 2, var, na.rm=TRUE),0.0)
pca_data <- pca_data[,drop_cols]


psaRes <- prcomp(pca_data, scale=TRUE, tol=0.1)

# what columns extracts
pcaCols <- sum(psaRes$sdev ^2 > 1)

# ok now from original set remove all columns in pca_data

orig <- as.data.frame(training_data$train)

# save id and sale price
orig_save <- as.data.frame(cbind(orig$Id,orig$SalePrice))
names(orig_save)[1] <- "Id"
names(orig_save)[2] <- "SalePrice"
col_fil <- !names(orig) %in% numeric_fields

# now for the false remove those columns
orig <- orig[,col_fil]

# and replace with the new x values
orig <- cbind(orig,psaRes$x[,1:pcaCols],SalePrice = orig_save$SalePrice)

varnames <- names(orig)
varnames <- varnames[!varnames %in% c('Id','SalePrice')]
varnames1 <- paste(varnames, collapse = "+")
form <- as.formula(paste("SalePrice",varnames1,sep = "~"))

set.seed(13)
# was 5000, depth 6 shrink .1 mnobs 10
gbm_model <- gbm(form, orig, distribution = "gaussian", n.trees = 5000,
                 train.fraction = 0.8, bag.fraction = 0.75, cv.folds = 5, 
                 interaction.depth = 6, n.minobsinnode=1)

summary(gbm_model) %>% filter(rel.inf>0.1) %>% ggplot() + geom_col(aes(reorder(var,rel.inf,descending=TRUE),rel.inf)) + coord_flip() + labs(title="Relative Influence", x="Variable")

gbm_perf <- gbm.perf(gbm_model, method = "cv")

# most important parts of PC1
attr(psaRes$rotation,"dimnames")[[1]][psaRes$rotation>0.1]

# ok lets drop some attributes
qe <- summary(gbm_model)
fields_of_i <- qe$var[qe$rel.inf >0]
varnames <- varnames[varnames %in% fields_of_i]
varnames1 <- paste(varnames, collapse = "+")
form <- as.formula(paste("SalePrice",varnames1,sep = "~"))
# #
set.seed(13)
# # # was 5000, depth 6 shrink .1 mnobs 10
gbm_model <- gbm(form, orig[,names(orig) %in% fields_of_i | names(orig) %in% c("SalePrice")], 
                   distribution = "gaussian", n.trees = 5000,
                   train.fraction = 0.8, bag.fraction = 0.75, cv.folds = 5, 
                   interaction.depth = 6, n.minobsinnode=1)
# #
summary(gbm_model) %>% filter(rel.inf>0.1) %>% ggplot() + geom_col(aes(reorder(var,rel.inf,descending=TRUE),rel.inf)) + coord_flip() + labs(title="Relative Influence", x="Variable")
# #
gbm_perf <- gbm.perf(gbm_model, method = "cv")
# 

# apply same to the trial data
##

pca_trial <- as.data.frame(training_data$trial)

# then this is only good for numerics
trial_data <- pca_trial[,factor_filter]

response_column2 <- which(names(trial_data) == 'SalePrice')

trial_data <- trial_data[,!(names(trial_data) %in% c("Id","SalePrice"))]


# now in this set constant columns could occur
drop_cols <- !near(apply(trial_data, 2, var, na.rm=TRUE),0.0)
trial_data <- trial_data[,drop_cols]

psaTrial <- prcomp(trial_data, scale=TRUE, tol=0.1)

# what columns extracts
pcaTrialCols <- sum(psaTrial$sdev ^2 > 1)

pcaCols == pcaTrialCols
# ok now from original set remove all columns in pca_data

origTrial <- as.data.frame(training_data$trial)

# save id and sale price
trial_save <- as.data.frame(cbind(origTrial$Id,origTrial$SalePrice))
names(trial_save)[1] <- "Id"
names(trial_save)[2] <- "SalePrice"
col_fil <- !names(origTrial) %in% numeric_fields

# now for the false remove those columns
origTrial <- origTrial[,col_fil]

# and replace with the new x values
#origTrial <- cbind(origTrial,psaTrial$x[,1:pcaCols],trial_save)
origTrial <- cbind(origTrial,psaTrial$x[,1:pcaCols])
#model_data <- origTrial[,names(origTrial) %in% fields_of_i]
##

# now what is the response so I can remove it
#trial_response_column <- which(names(origTrial) == 'SalePrice')
#newdata= origTrial[, -trial_response_column]
predictions_gbm <- predict(gbm_model, newdata= origTrial, 
                           n.trees = gbm_perf, type = "response")



print(localRMSE(predictions_gbm,trial_save$SalePrice))

ggplot() + geom_point(aes(trial_save$SalePrice,predictions_gbm),position="jitter")+geom_smooth(aes(trial_save$SalePrice,predictions_gbm))+geom_abline(slope=1,intercept=0,linetype="dashed",colour="red")+labs(title="Model Chart1")
ggsave("result1.png",width=16, height=7)
ggplot() + geom_point(aes(trial_save$SalePrice,predictions_gbm),position="jitter")+geom_smooth(aes(trial_save$SalePrice,predictions_gbm),method = "lm")+xlim(0,6e5)+ylim(0,6e5)+geom_abline(slope=1,intercept=0,linetype="dashed",colour="red")+labs(title="Model Chart 2")
ggsave("result2.png",width=16, height=7)

## ok try the test

test_data <- read_csv("test2.csv")
test_save_id <- test_data$Id


## Now clean

test_data <- clean_data(test_data)
test_save_data <- test_data

# recheck

absent_items <- sap_missing_values(test_data)
absent_items %>% ggplot() + geom_col(aes(reorder(feature,order(absent_items$gaps,decreasing = TRUE)),gaps,fill=gaps)) + coord_flip() + labs(title="Potential Data Gaps", x="Feature")

absent_items

response_column <- which(names(factor_filter) == 'SalePrice')
test_factor_filter <- factor_filter[-response_column]

test_data <- test_data[,test_factor_filter]

test_data <- test_data[,-1]


# now in this set constant columns can occur
drop_cols <- !near(apply(test_data, 2, var, na.rm=TRUE),0.0)
test_data <- test_data[,drop_cols]

psaTest <- prcomp(test_data, scale=TRUE, tol=0.1)



# what columns extracts
pcaTestCols <- sum(psaTest$sdev ^2 > 1)

pcaCols == pcaTestCols
# ok now from original set remove all columns in pca_data

origTest <- test_save_data

# save id 
col_fil <- !names(origTest) %in% numeric_fields

# now for the false remove those columns
origTest <- origTest[,col_fil]

# and replace with the new x values
origTest <- cbind(origTest,psaTest$x[,1:pcaCols])


# drop the columns not used
origTest <- origTest[,names(origTest) %in% fields_of_i]

##

predictions_gbm_test <- predict(gbm_model, newdata= origTest, 
                           n.trees = gbm_perf, type = "response")

r2 <- cbind("Id" = test_save_id,"SalePrice" = predictions_gbm_test)
write.csv(r2,"result.csv",row.names = FALSE)

print(r2)
















###### Now try xgboost with sparse matrix
library(xgboost)
library(Matrix)
library(data.table)

labels <- orig$SalePrice
origTrial2 <- cbind(origTrial,SalePrice=trial_save$SalePrice)
ts_label <- origTrial2$SalePrice

new_tr <- model.matrix(~.+0,data = orig[,-orig$SalePrice]) 
new_ts <- model.matrix(~.+0,data = origTrial2[,-origTrial2$SalePrice])

#convert factor to numeric 
labels <- as.numeric(labels)-1
ts_label <- as.numeric(ts_label)-1

dtrain <- xgb.DMatrix(data = new_tr,label = labels) 
dtest <- xgb.DMatrix(data = new_ts,label=ts_label)

bst <- xgboost(dtrain, max.depth = 4, eta = 1, nthread = 2, nround = 400)
pred <- predict(bst, dtest)

ggplot() + geom_point(aes(trial_save$SalePrice,pred),position="jitter")+geom_smooth(aes(trial_save$SalePrice,pred))+geom_abline(slope=1,intercept=0,linetype="dashed",colour="red")+labs(title="Model Chart1")
ggsave("result1_xgboost.png",width=16, height=7)

print(localRMSE(pred,trial_save$SalePrice))

### now the res


test_tr <- model.matrix(~.+0,data = origTest) 

dfntest <- xgb.DMatrix(data = test_tr)

pred2 <- predict(bst, dfntest)
r3 <- cbind("Id" = test_save_id,"SalePrice" = pred2)
write.csv(r3,"resultxgb.csv",row.names = FALSE)

