library(mice)

df <- read.csv("~/Desktop/aistats_2019/missing_data/data/df1.csv")
df <- subset(df, select = c(X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,outcome,treated,matched,true_effect))

miss <- read.csv("~/Desktop/aistats_2019/missing_data/data/miss1.csv")
miss <- subset(miss, select = c(X0,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14))

for(i in 1 : nrow(miss)){
  for(j in 1 : ncol(miss)){
    if(miss[i,j] == 1){
      df[i,j] <- NaN
    }
  }
}

imputed <- mice(df,m=10)
all_data <- complete(imputed, 'all')
for(i in 1 : 10){
  write.csv(all_data[[i]], file =paste("~/Desktop/aistats_2019/missing_data/data/imputed_", toString(i),".csv"))
}