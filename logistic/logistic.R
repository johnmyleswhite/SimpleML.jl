df <-  read.csv("logistic.csv", header = FALSE)
glm(V3 ~ V1 + V2 - 1,
	data = df,
	family = binomial(link = "logit"))
