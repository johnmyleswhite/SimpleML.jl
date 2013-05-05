df <- read.csv("linear.csv", header = FALSE)
lm(V3 ~ V1 + V2 - 1,
   data = df)
