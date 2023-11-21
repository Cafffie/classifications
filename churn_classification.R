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































#Checking the data set and converting categorical columns to factor
table(churning_data$gender)
churning_data$gender <- ifelse(churning_data$gender == "Female", "F", "M")
churning_data$gender <- as.factor(churning_data$gender)

table(churning_data$SeniorCitizen) 
churning_data$SeniorCitizen <- as.factor(churning_data$SeniorCitizen)

unique(churning_data$Dependents)
table(churning_data$Dependents)
churning_data$Dependents <- ifelse(churning_data$Dependents == "Yes", 1, 0)
churning_data$Dependents <- as.factor(churning_data$Dependents)

unique(churning_data$Partner)
churning_data$Partner <- ifelse(churning_data$Partner == "Yes", 1, 0)
churning_data$Partner <- as.factor(churning_data$Partner)

unique(churning_data$PhoneService)
churning_data$PhoneService <- ifelse(churning_data$PhoneService == "Yes", 1, 0)
churning_data$PhoneService <- as.factor(churning_data$PhoneService)

unique(churning_data$MultipleLines)
churning_data[churning_data$MultipleLines == "Yes",]$MultipleLines <- 1
churning_data[churning_data$MultipleLines == "No",]$MultipleLines <- 0
churning_data[churning_data$MultipleLines == "No phone service",]$MultipleLines <- 2
churning_data$MultipleLines <- as.factor(churning_data$MultipleLines)

unique(churning_data$InternetService)
churning_data[churning_data$InternetService == "No",]$InternetService <- 0
churning_data[churning_data$InternetService == "DSL",]$InternetService <- 1
churning_data[churning_data$InternetService == "Fiber optic",]$InternetService <- 2
churning_data$InternetService <- as.factor(churning_data$InternetService)

unique(churning_data$OnlineSecurity)
churning_data[churning_data$OnlineSecurity == "No",]$OnlineSecurity <- 0
churning_data[churning_data$OnlineSecurity == "Yes",]$OnlineSecurity <- 1
churning_data[churning_data$OnlineSecurity == "No internet service",]$OnlineSecurity <- 2
churning_data$OnlineSecurity <- as.factor(churning_data$OnlineSecurity)
      
unique(churning_data$OnlineBackup)
churning_data[churning_data$OnlineBackup == "No",]$OnlineBackup <- 0
churning_data[churning_data$OnlineBackup == "Yes",]$OnlineBackup <- 1
churning_data[churning_data$OnlineBackup == "No internet service",]$OnlineBackup <- 2
churning_data$OnlineBackup <- as.factor(churning_data$OnlineBackup)

unique(churning_data$DeviceProtection)
churning_data[churning_data$DeviceProtection == "No",]$DeviceProtection <- 0
churning_data[churning_data$DeviceProtection == "Yes",]$DeviceProtection <- 1
churning_data[churning_data$DeviceProtection == "No internet service",]$DeviceProtection <- 2
churning_data$DeviceProtection <- as.factor(churning_data$DeviceProtection)

unique(churning_data$TechSupport)
churning_data[churning_data$TechSupport == "No",]$TechSupport <- 0
churning_data[churning_data$TechSupport == "Yes",]$TechSupport <- 1
churning_data[churning_data$TechSupport == "No internet service",]$TechSupport <- 2
churning_data$TechSupport <- as.factor(churning_data$TechSupport)


unique(churning_data$StreamingTV)
churning_data[churning_data$StreamingTV == "No",]$StreamingTV <- 0
churning_data[churning_data$StreamingTV == "Yes",]$StreamingTV <- 1
churning_data[churning_data$StreamingTV == "No internet service",]$StreamingTV <- 2
churning_data$StreamingTV <- as.factor(churning_data$StreamingTV)

unique(churning_data$StreamingMovies)
churning_data[churning_data$StreamingMovies == "No",]$StreamingMovies <- 0
churning_data[churning_data$StreamingMovies == "Yes",]$StreamingMovies <- 1
churning_data[churning_data$StreamingMovies == "No internet service",]$StreamingMovies <- 2
churning_data$StreamingMovies <- as.factor(churning_data$StreamingMovies)

unique(churning_data$Contract)
churning_data[churning_data$Contract == "Month-to-month",]$Contract <- 3
churning_data[churning_data$Contract == "One year",]$Contract <- 1
churning_data[churning_data$Contract == "Two year",]$Contract <- 2
churning_data$Contract <- as.factor(churning_data$Contract)

unique(churning_data$PaperlessBilling)
churning_data$PaperlessBilling <- ifelse(churning_data$PaperlessBilling == "Yes", 1, 0)
churning_data$PaperlessBilling <- as.factor(churning_data$PaperlessBilling)


unique(churning_data$PaymentMethod)
churning_data[churning_data$PaymentMethod == "Electronic check",]$PaymentMethod <- 1
churning_data[churning_data$PaymentMethod == "Mailed check",]$PaymentMethod <- 2
churning_data[churning_data$PaymentMethod == "Bank transfer (automatic)",]$PaymentMethod <- 3
churning_data[churning_data$PaymentMethod == "Credit card (automatic)",]$PaymentMethod <- 4
churning_data$PaymentMethod <- as.factor(churning_data$PaymentMethod)


unique(churning_data$Churn)
churning_data$Churn <- ifelse(churning_data$Churn == "Yes", 1, 0)
churning_data$Churn <- as.factor(churning_data$Churn)

str(churning_data)



churn_yes <- churning_data[grepl("Yes", churning_data$SeniorCitizen), ]
dim(churn_yes)

logreg2 <- glm(Churn ~ tenure, data=churning_data, family="binomial")
summary(logreg1)

logreg <- glm(Churn ~., data=churning_data, family="binomial")
summary(logreg)

head(churning_data)

# Create a distribution plot for the continous variables
ggplot(data = churning_data, aes(x = tenure)) +
  geom_histogram(aes(y = ..density..), fill = "lightblue", color = "black", bins = 30) +
  geom_density(color = "blue") +
  geom_vline(xintercept = mean(churning_data$tenure), linetype = "dashed", color = "red", size = 1) +
  geom_text(x = 50, y = 0.03, label = paste("Density:", round(max(density(churning_data$tenure)$y), 3)))+
  labs(title = "Tenure")


ggplot(data = churning_data, aes(x = MonthlyCharges)) +
  geom_histogram(aes(y = ..density..), fill = "lightblue") +
  geom_density(color = "blue") +
  geom_vline(xintercept = mean(churning_data$MonthlyCharges), linetype = "dashed", color = "red") +
  geom_text(x = 50, y = 0.03, label = paste("Density:", round(max(density(churning_data$MonthlyCharges)$y), 3)))+
  labs(title = "MonthlyCharges")


ggplot(data = churning_data, aes(x = TotalCharges )) +
  geom_histogram(aes(y = ..density..), fill = "lightblue") +
  geom_density(color = "blue") +
  geom_vline(xintercept = mean(churning_data$TotalCharges ), linetype = "dashed", color = "red") +
  geom_text(x = 50, y = 0.03, label = paste("Density:", round(max(density(churning_data$TotalCharges)$y), 3))) +
  labs(title = "TotalCharges ")

#Total charges is right skewed
#performing a log to correct it
churning_data$TotalCharges <- log(churning_data$TotalCharges)

ggplot(data = churning_data, aes(x = TotalCharges )) +
  geom_histogram(aes(y = ..density..), fill = "lightblue") +
  geom_density(color = "blue") +
  geom_vline(xintercept = mean(churning_data$TotalCharges ), linetype = "dashed", color = "red") +
  geom_text(x = 50, y = 0.03, label = paste("Density:", round(max(density(churning_data$TotalCharges)$y), 3))) +
  labs(title = "TotalCharges ")

#Creating barplots for categorical variables
for (col in names(categorical_columns)) {
  barplot(table(churn2[[col]]), main= col)
}
colSums(is.na(churn2))
dim(churn2)

barplot(table(string_data$Gender), main = "Barplot of Gender", xlab = "Gender", ylab = "Frequency", col = "lightblue")
barplot(table(string_data$Education_Level), main = "Barplot of Education_Level", xlab = "Education_Level ", ylab = "Frequency", col = "lightblue")
barplot(table(string_data$Marital_Status), main = "Barplot of Marital_Status", xlab = "Marital_Status", ylab = "Frequency", col = "lightblue")
barplot(table(string_data$Income_Category), main = "Barplot of Income_Category", xlab = "Income_Category", ylab = "Frequency", col = "lightblue")
barplot(table(string_data$Card_Category), main = "Barplot of Card_Category", xlab = "Card_Category", ylab = "Frequency", col = "lightblue")
barplot(table(string_data$Attrition_Flag ), main = "Barplot of Attrition_Flag", xlab = "Attrition_Flag", ylab = "Frequency", col = "lightblue")




table(churning_data$gender)
table(churning_data$SeniorCitizen) 
table(churning_data$Partner)
table(churning_data$PhoneService)
table(churning_data$MultipleLines)
table(churning_data$InternetService)
table(churning_data$OnlineSecurity)
table(churning_data$OnlineBackup)
table(churning_data$DeviceProtection)
table(churning_data$TechSupport)
table(churning_data$StreamingTV)
table(churning_data$StreamingMovies)
table(churning_data$Contract)
table(churning_data$PaperlessBilling)
table(churning_data$PaymentMethod)
table(churning_data$Churn)
table(churning_data$SeniorCitizen)
table(churning_data$SeniorCitizen)
table(churning_data$SeniorCitizen)












