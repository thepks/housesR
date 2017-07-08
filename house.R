library(readr)
library(purrr)
library(dplyr)
library(Amelia)
library(tidyr)
library(ggplot2)
library(stringr)
library(gbm)
library(forcats)
library(caret)
library(GGally)
library(dummies)

house_raw <- read_csv('train.csv')

# number of houses
nrow(house_raw)

#number of attributes
ncol(house_raw)-1

#alter names to start with char
# alter the name 
mod_name <- function(f) { if(!is.na(str_match(f,"^[0-9].*"))){paste0("val_",f)} else {f}}
names(house_raw) <- lapply(names(house_raw),mod_name)



# check columns for na
# check columns for na
check_na <- function(f) {sum(is.na(f))}

absent_data_cols <- summarise_all(house_raw,check_na)
absent_data_cols_ind <- absent_data_cols[1,]>0
absent_items <- t(absent_data_cols[,absent_data_cols_ind[1,]])
absent_items <- cbind(absent_items,feature=row.names(absent_items))
absent_items <- as.data.frame(absent_items,stringsAsFactors=FALSE) 
names(absent_items)[1] <- "gaps"
absent_items$gaps <- as.numeric(absent_items$gaps)
absent_items <- absent_items[order(absent_items$gaps,decreasing = TRUE),]
row.names(absent_items) <- NULL
absent_items %>% ggplot() + geom_col(aes(reorder(feature,order(absent_items$gaps,decreasing = TRUE)),gaps,fill=gaps)) + coord_flip() + labs(title="Potential Data Gaps", x="Feature")

# Set some of the absents to None


house_raw[is.na(house_raw$PoolQC),]$PoolQC <- 'None'
house_raw[is.na(house_raw$Alley),]$Alley <- 'None'
house_raw[is.na(house_raw$Fence),]$Fence <- 'None'
house_raw[is.na(house_raw$FireplaceQu),]$FireplaceQu <- 'None'

house_no_gar <- house_raw$GarageArea <1

house_raw[house_no_gar,]$GarageCond <- "None"
house_raw[house_no_gar,]$GarageQual <- "None"
house_raw[house_no_gar,]$GarageFinish <- "None"
house_raw[house_no_gar,]$GarageYrBlt <- "None"
house_raw[house_no_gar,]$GarageType <- "None"

house_no_bsmt <- house_raw$BsmtUnfSF < 1
house_raw[house_no_bsmt,]$BsmtFinType1 <- "None"
house_raw[house_no_bsmt,]$BsmtFinType2 <- "None"
house_raw[house_no_bsmt,]$BsmtExposure <- "None"
house_raw[house_no_bsmt,]$BsmtCond <- "None"
house_raw[house_no_bsmt,]$BsmtQual <- "None"

house_raw[is.na(house_raw$MiscFeature),]$MiscFeature <- 'None'

house_raw[is.na(house_raw$LotFrontage),]$LotFrontage <- 0
house_raw[is.na(house_raw$Electrical),]$Electrical <- 0

# sort out the factors
fields_class <- sapply(house_raw,class)
char_fields <- fields_class[fields_class == "character"]
house_char <- house_raw
house_raw[names(char_fields)] <- lapply(house_raw[names(char_fields)],factor)

# recheck

absent_data_cols <- summarise_all(house_raw,check_na)
absent_data_cols_ind <- absent_data_cols[1,]>0
absent_items <- t(absent_data_cols[,absent_data_cols_ind[1,]])
absent_items <- cbind(absent_items,feature=row.names(absent_items))
absent_items <- as.data.frame(absent_items,stringsAsFactors=FALSE) 
names(absent_items)[1] <- "gaps"
absent_items$gaps <- as.numeric(absent_items$gaps)
absent_items <- absent_items[order(absent_items$gaps,decreasing = TRUE),]
row.names(absent_items) <- NULL
absent_items %>% ggplot() + geom_col(aes(reorder(feature,order(absent_items$gaps,decreasing = TRUE)),gaps,fill=gaps)) + coord_flip() + labs(title="Potential Data Gaps", x="Feature")



# split the data into 3 sets 60:20:20


inTrain <- createDataPartition(house_raw$Id, p=0.6, times=1)
training_data <- house_raw[inTrain$Resample1,]
training_test_data <- house_raw [ - inTrain$Resample1,]

response_column <- which(names(house_raw) == 'SalePrice')
varnames <- names(training_data)
varnames <- varnames[!varnames %in% c('Id','SalePrice')]
varnames1 <- paste(varnames, collapse = "+")
form <- as.formula(paste("SalePrice",varnames1,sep = "~"))

set.seed(13)
gbm_model <- gbm(form, training_data, distribution = "gaussian", n.trees = 3000, 
                 bag.fraction = 0.75, cv.folds = 5, interaction.depth = 3, n.minobsinnode=1)

summary(gbm_model) %>% filter(rel.inf>0.1) %>% ggplot() + geom_col(aes(reorder(var,rel.inf,descending=TRUE),rel.inf)) + coord_flip() + labs(title="Relative Influence", x="Variable")

gbm_perf <- gbm.perf(gbm_model, method = "cv")

predictions_gbm <- predict(gbm_model, newdata= training_test_data[, -response_column], 
                           n.trees = gbm_perf, type = "response")

RMSE(predictions_gbm,training_test_data$SalePrice)

ggplot() + geom_point(aes(training_test_data$SalePrice,predictions_gbm),position="jitter")+geom_smooth(aes(training_test_data$SalePrice,predictions_gbm))

ggplot() + geom_point(aes(training_test_data$SalePrice,predictions_gbm),position="jitter")+geom_smooth(aes(training_test_data$SalePrice,predictions_gbm),method = "lm")+xlim(0.75e5,3e5)


# try one hot encoding of all the character fields rather than factors
# this is going to imapact these fields:
fc1 <- which(sapply(house_char,class) == "character")
print(paste(names(house_raw)[fc1],sep = " "))

house_dummy <- dummy.data.frame(house_char)
# now form a formula