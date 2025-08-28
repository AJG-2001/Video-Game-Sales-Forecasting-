##############################################################################
### Assignment A3: Video Game Sales – Business Case Modeling and Forecast ####
### Goal: Predict success of video games and forecast future sales ###########
### Student Name: Abi Joshua George (46656697) ###############################
##############################################################################

# --- Load Required Libraries ---
library(readr)
library(dplyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)
library(tseries)
library(forecast)

# --------------------------------|
# STEP 1: LOAD AND INITIAL CLEAN  |
# --------------------------------|

df <- read_csv("C:/Users/abijo/OneDrive/Desktop/Assignment A3/Video_Games_Sales.csv")
summary(df)
names(df)

# -----------------------------------------|
# STEP 2: CLEANING AND FEATURE ENGINEERING |
# -----------------------------------------|

# Remove rows with NA Year or Global_Sales
df_clean <- df %>%
  filter(!is.na(Year), !is.na(Global_Sales), Global_Sales > 0)

# Convert Year to numeric
df_clean$Year <- as.numeric(df_clean$Year)

# Business outcome: Success if global sales > 0.5M units
df_clean$business_outcome <- ifelse(df_clean$Global_Sales > 0.5, 1, 0)

# Normalize numeric sales columns
rescale <- function(x) (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
df_clean$NA_Sales_norm     <- rescale(df_clean$NA_Sales)
df_clean$EU_Sales_norm     <- rescale(df_clean$EU_Sales)
df_clean$JP_Sales_norm     <- rescale(df_clean$JP_Sales)
df_clean$Other_Sales_norm  <- rescale(df_clean$Other_Sales)
df_clean$Global_Sales_norm <- rescale(df_clean$Global_Sales)

# Convert categorical columns
df_clean$Genre     <- as.factor(df_clean$Genre)
df_clean$Platform  <- as.factor(df_clean$Platform)
df_clean$Publisher <- as.factor(df_clean$Publisher)

summary(df_clean)

# -------------------------------|
# STEP 3: TRAIN-TEST SPLIT       |
# -------------------------------|

set.seed(123)
indx <- sample(1:nrow(df_clean), size = 0.8 * nrow(df_clean))
game_train <- df_clean[indx, ]
game_test  <- df_clean[-indx, ]

# --------------------------------------------|
# Framework 1A: Decision Tree – RAW VARIABLES |
# --------------------------------------------|

my_tree <- rpart(business_outcome ~ NA_Sales + JP_Sales + EU_Sales + Other_Sales,
                 data = game_train, method = "class", cp = 0.01)
rpart.plot(my_tree)

tree_pred <- predict(my_tree, game_test)
confusionMatrix(
  data = factor(as.numeric(tree_pred[, 2] > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(game_test$business_outcome), levels = c(0, 1))
)

# ---------------------------------------------------|
# Framework 1B: Decision Tree – NORMALIZED VARIABLES |
# ---------------------------------------------------|

my_tree_norm <- rpart(business_outcome ~ NA_Sales_norm + JP_Sales_norm + EU_Sales_norm + Other_Sales_norm,
                      data = game_train, method = "class", cp = 0.01)
rpart.plot(my_tree_norm)

tree_pred_norm <- predict(my_tree_norm, game_test)
confusionMatrix(
  data = factor(as.numeric(tree_pred_norm[, 2] > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(game_test$business_outcome), levels = c(0, 1))
)

# --------------------------------------------|
# Framework 2A: Random Forest – RAW VARIABLES |
# --------------------------------------------|

my_forest <- randomForest(as.factor(business_outcome) ~ NA_Sales + EU_Sales + JP_Sales + Other_Sales,
                          data = game_train, ntree = 100, mtry = 2, importance = TRUE)
varImpPlot(my_forest)

forest_pred <- predict(my_forest, game_test)
confusionMatrix(data = forest_pred, reference = factor(game_test$business_outcome))

# ---------------------------------------------------|
# Framework 2B: Random Forest – NORMALIZED VARIABLES |
# ---------------------------------------------------|

my_forest_norm <- randomForest(as.factor(business_outcome) ~ NA_Sales_norm + EU_Sales_norm +
                                 JP_Sales_norm + Other_Sales_norm,
                               data = game_train, ntree = 100, mtry = 2, importance = TRUE)
varImpPlot(my_forest_norm)

forest_pred_norm <- predict(my_forest_norm, game_test)
confusionMatrix(data = forest_pred_norm, reference = factor(game_test$business_outcome))

# ---------------------------------------------|
# Framework 3A: Neural Network – RAW VARIABLES |
# ---------------------------------------------|

my_neural_raw <- neuralnet(business_outcome ~ NA_Sales + EU_Sales + JP_Sales + Other_Sales,
                           data = game_train, hidden = c(4, 2), linear.output = FALSE)
plot(my_neural_raw, rep = "best")

neural_pred_raw <- predict(my_neural_raw, game_test)
confusionMatrix(
  data = factor(as.numeric(neural_pred_raw > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(game_test$business_outcome), levels = c(0, 1))
)

# ----------------------------------------------------------------------------|
# Framework 3B: Neural Network – NORMALIZED VARIABLES [TAKES TOO LONG TO RUN] |
# ----------------------------------------------------------------------------|

my_neural_norm <- neuralnet(business_outcome ~ NA_Sales_norm + EU_Sales_norm +
                              JP_Sales_norm + Other_Sales_norm,
                            data = game_train, hidden = c(3, 2), linear.output = FALSE, stepmax = 1e6)
plot(my_neural_norm, rep = "best")

neural_pred_norm <- predict(my_neural_norm, game_test)
confusionMatrix(
  data = factor(as.numeric(neural_pred_norm > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(game_test$business_outcome), levels = c(0, 1))
)

# --------------------------------------------------|
# Framework 4A: Logistic Regression – RAW VARIABLES |
# --------------------------------------------------|

logit_model_raw <- glm(business_outcome ~ NA_Sales + EU_Sales + JP_Sales + Other_Sales,
                       data = game_train, family = binomial)
logit_pred_raw <- predict(logit_model_raw, game_test, type = "response")
confusionMatrix(
  data = factor(as.numeric(logit_pred_raw > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(game_test$business_outcome), levels = c(0, 1))
)

# ---------------------------------------------------------|
# Framework 4B: Logistic Regression – NORMALIZED VARIABLES |
# ---------------------------------------------------------|

logit_model_norm <- glm(business_outcome ~ NA_Sales_norm + EU_Sales_norm +
                          JP_Sales_norm + Other_Sales_norm,
                        data = game_train, family = binomial)
logit_pred_norm <- predict(logit_model_norm, game_test, type = "response")
confusionMatrix(
  data = factor(as.numeric(logit_pred_norm > 0.5), levels = c(0, 1)),
  reference = factor(as.numeric(game_test$business_outcome), levels = c(0, 1))
)

# ---------------------------------------------------------|
# Time Series Forecasting – Descriptive Statistics & Plots |
# ---------------------------------------------------------|

# Descriptive Statistics
summary(df_clean[, c("NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales")])

# Histogram of Global Sales
ggplot(df_clean, aes(x = Global_Sales)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Global Video Game Sales",
       x = "Global Sales (millions)", y = "Number of Games")

# Year-wise average sales
df_clean <- df_clean %>%
  filter(Year >= 1980 & Year <= 2025)

df_clean %>%
  group_by(Year) %>%
  summarise(avg_sales = mean(Global_Sales, na.rm = TRUE)) %>%
  ggplot(aes(x = Year, y = avg_sales)) +
  geom_line(color = "darkgreen") +
  labs(title = "Average Global Sales by Year",
       x = "Year", y = "Avg Global Sales (millions)")

# Sales by Genre
ggplot(df_clean, aes(x = Genre, y = Global_Sales)) +
  geom_boxplot(fill = "coral") +
  coord_flip() +
  labs(title = "Global Sales by Genre", x = "Genre", y = "Global Sales (millions)")

# Average sales by Platform
df_clean %>%
  group_by(Platform) %>%
  summarise(avg_sales = mean(Global_Sales, na.rm = TRUE)) %>%
  ggplot(aes(x = reorder(Platform, avg_sales), y = avg_sales)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Average Global Sales by Platform",
       x = "Platform", y = "Average Global Sales (millions)")

# Summarize total global sales per publisher
publisher_sales <- df_clean %>%
  group_by(Publisher) %>%
  summarise(Total_Global_Sales = sum(Global_Sales, na.rm = TRUE)) %>%
  arrange(desc(Total_Global_Sales)) %>%
  top_n(10, Total_Global_Sales)

# Bar plot of top 10 publishers by total global sales
library(ggplot2)
ggplot(publisher_sales, aes(x = reorder(Publisher, Total_Global_Sales), y = Total_Global_Sales)) +
  geom_bar(stat = "identity", fill = "darkorange") +
  coord_flip() +
  labs(title = "Top 10 Publishers by Total Global Sales",
       x = "Publisher", y = "Total Global Sales (millions)")


# --------------------------------------------------|
# Stationarity Checks: ADF, ACF, PACF – Time Series |
# --------------------------------------------------|

avg_sales_yearly <- df_clean %>%
  group_by(Year) %>%
  summarise(avg_global_sales = mean(Global_Sales, na.rm = TRUE)) %>%
  filter(!is.na(Year) & Year >= 1980 & Year <= 2020)

avg_sales_ts <- ts(avg_sales_yearly$avg_global_sales, start = 1980, frequency = 1)
diff_sales_ts <- diff(avg_sales_ts)

adf.test(diff_sales_ts)
acf(diff_sales_ts, main = "ACF: Differenced Global Sales")
pacf(diff_sales_ts, main = "PACF: Differenced Global Sales")

# -------------------------------------|
# Framework 5: ARIMA Forecasting Model |
# -------------------------------------|

model_arima <- auto.arima(avg_sales_ts)
summary(model_arima)

forecast_arima <- forecast(model_arima, h = 5)
plot(forecast_arima, main = "ARIMA Forecast – Global Sales")

