library(caret)
library(kernlab)
library(dummy)
library(fastDummies)
library(ROCR)
library(pROC)
library(ROSE)
library(xgboost)
library(e1071)
library(randomForest)


#Loading the dataset
churning_data <- read.csv("./data_Science/churning.csv")
churn2<- churning_data

#Checking the top 5 rows
churning_data[0:5,]

#Exploring the data 
str(churning_data)
dim(churning_data)
summary(churning_data)

#Checking for missing values
colSums(is.na(churning_data))
nrow(churning_data[is.na(churning_data$TotalCharges),])

#Viewing the rows that contains na
churning_data[is.na(churning_data$TotalCharges),]

#Handling missing values (replacing with the mean)
churning_data$TotalCharges[is.na(churning_data$TotalCharges)] <- mean(churning_data$TotalCharges, na.rm=TRUE)
churn2$TotalCharges[is.na(churn2$TotalCharges)] <- mean(churn2$TotalCharges, na.rm=TRUE)

head(churn2)

#Drop unneeded column
churn2 <- churn2[, -1]
str(churn2)

#Select categorical columns
categorical_columns <- churn2 %>%
  select_if(~ !is.numeric(.))
head(categorical_columns)
dim(categorical_columns)

#Select numeric columns
numeric_columns <- churn2 %>%
  select_if(is.numeric)
head(numeric_columns)
dim(numeric_columns)
colSums(is.na(churn2))  

#Viewing unique values in categorical columns
for (col in names(categorical_columns)) {
  print(paste(col, ':', toString(unique(churn2[[col]]))))
}

for (col in names(categorical_columns)){
  print(paste(col, ':', unique(categorical_columns[[col]])))
}

#Replacing values in categorical columns
for (col in names(categorical_columns)) {
  churn2[[col]] <- gsub("No internet service", "No", churn2[[col]])
  churn2[[col]] <- gsub("No phone service", "No", churn2[[col]])
}

#Encoding text in categorical columns to number
for (col in names(categorical_columns)){
  churn2[[col]]  <- gsub("Yes", 1, churn2[[col]])
  churn2[[col]] <- gsub("No", 0, churn2[[col]])
  churn2[[col]] <- gsub("Female", 1, churn2[[col]])
  churn2[[col]] <- gsub("Male", 0, churn2[[col]])
}
churn2$InternetService <- gsub(0, "No", churn2$InternetService)
#churn_with_hyphen <- churn[grepl("-", churn$LAST_12_MONTHS_CREDIT_VALUE), ]

#Creating dummy variables for categorical features
churn2 <- dummy_cols(churn2, select_columns = c("InternetService", "Contract", "PaymentMethod"), remove_first_dummy = TRUE)
head(churn2)
head(churn3)

#Dropping repeated columns
churn2 <- churn2[, !names(churn2) %in% c("InternetService", "Contract", "PaymentMethod")]
dim(churn2)
str(churn2)

#Converting all numeric variables to numeric
for (col in names(churn2)){
  if (class(churn2[[col]]) == "character"){
    churn2[[col]] <- as.numeric(churn2[[col]])
  }
}

churn3 <- churn2
write.csv(churn3, file= "./data_Science/dummy_churn.csv")

#logging the skewed variables to make the distribution normal
#churn2$TotalCharges <- log(churn2$TotalCharges)
#churn2$tenure <- log(churn2$tenure)


#Scaling the numeric variables
#columns <- c("tenure", "MonthlyCharges", "TotalCharges")
##  churn2[[col]] <- scale(churn2[[col]])
#}

head(churn2)
str(churn2)


#UNIVARIATE ANALYSIS
#Plotting barplots for all categorical columns
for (column in names(categorical_columns)) {
  barplot(table(categorical_columns[[column]]), main= column, col="pink")
}
barplot(table(churn2$SeniorCitizen), main= SeniorCitizen, col="black")

#Distribution plot for continuous variables
#Tenure
plot1 <- ggplot(churn2, aes(x = tenure)) +
  geom_histogram(aes(y= ..density..), fill = "blue") +
  geom_density(color = "red") +
  geom_vline(aes(xintercept = mean(tenure)), color = "black", linetype = "dashed") +
  labs(title = "Tenure", x = "Tenure")
ggplotly(plot1)

#MonthlyCharges
plot2 <- ggplot(churn2, aes(x = MonthlyCharges)) +
  geom_histogram(aes(y= ..density..), fill = "blue") +
  geom_density(color = "red") +
  geom_vline(aes(xintercept = mean(MonthlyCharges)), color = "black", linetype = "dashed") +
  labs(title = "MonthlyCharges", x = "MonthlyCharges")
ggplotly(plot2)

#TotalCharges
plot3 <- ggplot(churn2, aes(x = TotalCharges)) +
  geom_histogram(aes(y= ..density..), fill = "blue") +
  geom_density(color = "red") +
  geom_vline(aes(xintercept = mean(TotalCharges)), color = "black", linetype = "dashed") +
  labs(title = "TotalCharges", x = "TotalCharges")
ggplotly(plot3)



#The data seems imbalanced.
#Using SMOTE to correct it
#Build the model
log1 <- glm(Churn ~ ., data = churn2, family = "binomial")
summary(log1)
log1$fitted.values

#Split the data set into training and test set
churn2$Churn <- as.factor(churn2$Churn)
intrain <- createDataPartition(y=churn2$Churn, p=0.75, list=FALSE)
train <- churn2[intrain,]
test <- churn2[-intrain,]

#Build logistic regression model
log2 <- train(Churn ~., data=train, method="glm", family="binomial")
summary(log2)

#Model prediction
predictions1 <- predict(log2, newdata=test)

#Model accuracy
accuracy1 <- confusionMatrix(predictions1, test$Churn)
print(accuracy1)
Accuracy : 0.8102 
Recall1 <- 0.8574578
#Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))





#Build a random forest model
library(randomForest)
rf <- train(Churn ~., data=train, method= "rf", 
            trControl=trainControl(method="cv", number=5))
print(rf)

#Predict with the random forest model
predictions2 <- predict(rf, newdata=test)

#Evaluate the model
accuracy2 <- confusionMatrix(predictions2,test$Churn)
print(paste("Accuracy for Random Forest Model:", accuracy2$overall["Accuracy"]))
Accuracy : 0.8176  
recall2 <- 0.8384401

  


#Build a support vector model
library(e1071)
svm <- train(Churn ~., data=train, method= "svmRadial",
             trControl=trainControl(method="cv", number=5))
print(svm)

#Perform SVM prediction
predictions3 <- predict(svm, newdata=test)
#Evaluate the model
accuracy3 <- confusionMatrix(predictions3, test$Churn)
print(paste("Accuracy for svmRadial:", accuracy3$overall["Accuracy"]))
Accuracy : 0.8176
recall3 <- 0.8437058
  
  
  
  

svm2 <- train(Churn ~., data=train, method="svmLinear",
              trControl= trainControl(method="cv", number=5))
print(svm2)
#Perform SVM prediction
predictions4 <- predict(svm2, newdata=test)

#Evaluate the model
accuracy4 <- confusionMatrix(predictions4, test$Churn)
print(paste("Accuracy for svmLinear:", accuracy4$overall["Accuracy"]))
Accuracy : 0.8125
recall4 <- 0.8522312





#Build a K-Nearest Neighbor model
knn <- train(Churn ~., data=train, method="knn",
             trControl=trainControl(method="cv", number = 5),
             tuneGrid= expand.grid(k= c(1, 3, 5))) #specify the "k" values to try

print(knn)
#Model prediction
predictions5 <- predict(knn, test)

#Evaluate the KNN model
accuracy5 <- confusionMatrix(predictions5, test$Churn)
print(paste("Accuracy for KNN model:", accuracy5$overall["Accuracy"]))
Accuracy : 0.7688
Recall5 <- 0.8177905




library(xgboost)
#Build the xgboost model
xgboost <- train(Churn ~., data=train, method= "xgbTree",
                 trControl=trainControl(method= "cv", number = 5))
print(xgboost)

#Make predictions
predictions6 <- predict(xgboost, test)

#Evaluate the model
accuracy6 <- confusionMatrix(predictions6, test$Churn)
Accuracy : 0.8102
Recall6 <- 0.8412811




#Comparing the performance of the six models
all_models <- data.frame(
  models = c("LOGR", "RandomF", "SVMr", "SVMl", "KNN", "XGboost"),
  Recall= c(0.8574578, 0.8384401, 0.8437058, 0.8522312, 0.8177905, 0.8412811),
  accuracy = c(accuracy1$overall["Accuracy"], 
               accuracy2$overall["Accuracy"],
               accuracy3$overall["Accuracy"],
               accuracy4$overall["Accuracy"],
               accuracy5$overall["Accuracy"],
               accuracy6$overall["Accuracy"]))

Recall= c(toString(Recall1), toString(Recall3), toString(Recall4), toString(Recall5), toString(Recall6))

all_models
models    Recall  accuracy
1    LOGR 0.8574578 0.8181818
2 RandomF 0.8384401 0.8176136
3    SVMr 0.8437058 0.8176136
4    SVMl 0.8522312 0.8125000
5     KNN 0.8177905 0.7687500
6 XGboost 0.8412811 0.8102273
>































