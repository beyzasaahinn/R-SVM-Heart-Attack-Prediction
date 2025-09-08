library(caret)
library(e1071)
library(kernlab)
library(dplyr)
library(ggplot2)
library(pROC)
library(recipes)
library(themis)  # SMOTE için

heart <- read.csv("~/Desktop/03403403/heart_attack_prediction_dataset.csv")

head(heart)
str(heart)
summary(heart)

colSums(is.na(heart))

heart$Heart.Attack.Risk <- as.factor(heart$Heart.Attack.Risk)

table(heart$Heart.Attack.Risk)
prop.table(table(heart$Heart.Attack.Risk))

heart <- heart %>% select(-Patient.ID, -Country, -Continent, -Hemisphere)
head(heart)

ggplot(heart, aes(x = Heart.Attack.Risk, y = Age, fill = Heart.Attack.Risk)) +
  geom_boxplot() +
  labs(title = "Age vs Heart Attack Risk", y = "Age", x = "Risk") +
  theme_minimal()

ggplot(heart, aes(x = Heart.Attack.Risk, y = BMI, fill = Heart.Attack.Risk)) +
  geom_boxplot() +
  labs(title = "BMI vs Heart Attack Risk", y = "BMI", x = "Risk") +
  theme_minimal()

set.seed(123)

train_index <- createDataPartition(heart$Heart.Attack.Risk, p = 0.7, list = FALSE)
train_data <- heart[train_index, ]
test_data <- heart[-train_index, ]

rec <- recipe(Heart.Attack.Risk ~ ., data = train_data) %>%
  step_mutate(BP_Systolic = as.numeric(sub("/.*", "", Blood.Pressure)),
              BP_Diastolic = as.numeric(sub(".*/", "", Blood.Pressure))) %>%
  step_rm(Blood.Pressure) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  step_smote(Heart.Attack.Risk)

tune_grid <- expand.grid(
  sigma = 2^seq(-15, 3, by = 3),  # daha geniş sigma 
  C = 2^seq(-3, 5, by = 2)        # daha geniş C 
)

ctrl <- trainControl(method = "cv", number = 10)

set.seed(123)
svm_grid_model <- train(rec,
                        data = train_data,
                        method = "svmRadial",
                        trControl = ctrl,
                        metric = "Accuracy",
                        tuneGrid = tune_grid)

print(svm_grid_model)
plot(svm_grid_model)

pred <- predict(svm_grid_model, newdata = test_data)
conf <- confusionMatrix(pred, test_data$Heart.Attack.Risk, positive = "1")
print(conf)
