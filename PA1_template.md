# Human Activity Recognition


```r
library(caret)
library(randomForest)
library(parallel)
library(doParallel)
```
## Synopsis
In this paper we will use the Human Activity Recognition (HAR) dataset to predict how well a subject did an exercise.
This is a multiclass classigication problem in which we will perform the folowing steps:

* Initial data analysis and pre-processing
* Data splitting into training and testing sets
* Random-Forest training on 3-Fold Cross-Validation
* Scoring and validation on the validation set.

## Initial Analysis and Pre-Processing
Loading the data we see many empty fieldsand fields with values of "#DIV/0!".
We will consider these values as NA.

```r
train <- read.csv("C:/Users/tlevy/Desktop/Tal/Opera/Courses/JHU Data Science/scripts/HAR/pml-training.csv", na.strings = c("#DIV/0!","NA",""))
test <- read.csv("C:/Users/tlevy/Desktop/Tal/Opera/Courses/JHU Data Science/scripts/HAR/pml-testing.csv", na.strings = c("#DIV/0!","NA",""))

dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```

```r
#summary(train)
#table(train$classe)
```
Our data consists of 160 columns.
Our first step will be removing columns with not enough available data.
Columns with over 70% null values will be removed.
In addition, the first seven columns in the dataset give us information that is redundant for modeling (index of sample, user name and five time features) and will be removed as well.

```r
nulls<-(colSums(is.na(train))/nrow(train))>0.7
train <- train[!nulls]
test <- test[!nulls]
# names(train[(colSums(is.na(train))/nrow(train))<0.7])
train <- train[,-(1:7)]
test <- test[,-(1:7)]
```
Next step will be a more analytical feature selection.
We will remove features that do not add additional information.
We will perform pairwise correlation and remove the columns with over 75% pairwise correlation.

```r
correlationMatrix<-cor(train[-53])
highCorrelation <- findCorrelation(correlationMatrix, cutoff=0.75)
train <- train[-highCorrelation]
test <- test[-highCorrelation]

dim(train)
```

```
## [1] 19622    32
```

```r
dim(test)
```

```
## [1] 20 32
```
We are left with 32 variables and our dataset is now 1/5 the size of the original set.
## Data Splitting and Training
We will start the training process by splitting our data to 70% training and 30% validation.

```r
set.seed(12345)
inTrain <- createDataPartition(y=train$classe,p=0.7,list = FALSE)
training <- train[inTrain,]
validation <- train[-inTrain,]
```
We will use 3-Fold Cross Validation on the training set and train using Random-Forest method.
For faster training, we will use parallel processing.

```r
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv",
                           number = 3,
                           allowParallel = TRUE)
modRF<-train(classe~.,data = training, method = "rf", trControl=fitControl)
stopCluster(cluster)
registerDoSEQ()
```
Looking at the created model we see very good results:

* All 3 folds performed well with accuracy over 97.7%
* Out Of Sample error rate of chosen model is as low as 0.87%
* Average accuracy is over 98%


```r
modRF
```

```
## Random Forest 
## 
## 13737 samples
##    31 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9158, 9159, 9157 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9868970  0.9834223
##   16    0.9858779  0.9821341
##   31    0.9785987  0.9729238
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
modRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.87%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3900    2    1    2    1 0.001536098
## B   19 2631    7    0    1 0.010158014
## C    0   28 2352   15    1 0.018363940
## D    0    1   34 2214    3 0.016873890
## E    0    0    0    5 2520 0.001980198
```

```r
modRF$resample
```

```
##    Accuracy     Kappa Resample
## 1 0.9842795 0.9801090    Fold3
## 2 0.9884229 0.9853569    Fold2
## 3 0.9879886 0.9848011    Fold1
```

```r
confusionMatrix.train(modRF)
```

```
## Cross-Validated (3 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.2  0.0  0.0  0.0
##          B  0.0 18.9  0.3  0.0  0.0
##          C  0.0  0.1 17.0  0.3  0.0
##          D  0.0  0.0  0.1 16.0  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9869
```
## Validation
our final step will be checking performance on the validation set

```r
validation.predict <- predict(modRF,validation)
confusionMatrix(validation.predict,validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672   11    0    0    0
##          B    2 1120   18    0    0
##          C    0    7 1000   23    0
##          D    0    0    8  941    3
##          E    0    1    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9876          
##                  95% CI : (0.9844, 0.9903)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9843          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9833   0.9747   0.9761   0.9972
## Specificity            0.9974   0.9958   0.9938   0.9978   0.9998
## Pos Pred Value         0.9935   0.9825   0.9709   0.9884   0.9991
## Neg Pred Value         0.9995   0.9960   0.9946   0.9953   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1903   0.1699   0.1599   0.1833
## Detection Prevalence   0.2860   0.1937   0.1750   0.1618   0.1835
## Balanced Accuracy      0.9981   0.9896   0.9842   0.9870   0.9985
```
At 98.7%, accuracy on the validation set is also very high proving our model to be very efficient.

## Conclusions 
* While the initial dataset had many variables, ony 1/5 was needed for building an accurate model.
* Random-Forest model using 3-Fold Cross-Validation was proven to give very high accuracy.
* The OOS estimate of error rate was very close to the validation error.

## Appendix A
### Course Project Prediction Quiz Portion
The following is the model prediction on the test set

```r
test.predict <- predict(modRF,test)
test.predict
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
submisison <- data.frame(test$problem_id,test.predict)
```
