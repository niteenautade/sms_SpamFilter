smsdata <- read.csv("sms_spam.csv", stringsAsFactors = FALSE)
str(smsdata)
smsdata$type <- factor(smsdata$type)
str(smsdata)
table(smsdata$type)
install.packages("tm")
library(tm)
sms_corpus <- Corpus(VectorSource(smsdata$text))
print(sms_corpus)
corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

sms_dtm <- DocumentTermMatrix(corpus_clean)

smsTrain<- smsdata[1:4169,]
smsTest<- smsdata[4170:5559,]

sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

sms_corpus_train <- corpus_clean[1:4169]
sms_corpus_test <- corpus_clean[4170:5559]

prop.table(table(smsTrain$type))
prop.table(table(smsTest$type))

install.packages("wordcloud")
library(wordcloud)

wordcloud(sms_corpus_train, min.freq = 40, random.order = FALSE)

spam <- subset(smsTrain, type == "spam")
ham <- subset(smsTrain, type == "ham")

wordcloud(spam$text, max.words = 40, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 40, scale = c(3, 0.5))

sms_dict <- c(findFreqTerms(sms_dtm_train, 5))

sms_train <- DocumentTermMatrix(sms_corpus_train,list(dictionary = sms_dict))
sms_test <- DocumentTermMatrix(sms_corpus_test,list(dictionary = sms_dict))

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}

sms_train <- apply(sms_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_test, MARGIN = 2, convert_counts)

library(e1071)

sms_classifier <- naiveBayes(sms_train, smsTrain$type)
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, smsTest$type,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual'))

sms_classifier2 <- naiveBayes(sms_train, smsTrain$type,
                              laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)

CrossTable(sms_test_pred2, smsTest$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
