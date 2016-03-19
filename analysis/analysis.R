data = read.table("../data/F_data_oil_SUPAERO.csv", header = F, sep = ",", dec = ".")
data_ = data.frame(data)

label = read.table("../data/F_label_oil_SUPAERO.csv", header = F, sep = ",", dec = ".")
colnames(label) = c('label')


data$label = label

data = data.frame(data)

data_0 = data[data$label == 0, ];
data_1 = data[data$label == 1, ];

data_0_ = data.frame(data_0)
data_1_ = data.frame(data_1)

data_0_$label = NULL
data_1_$label = NULL

nb_0 = nrow(data_0)
nb_1 = nrow(data_1)
total = nb_0 + nb_1

nb_0
nb_1
total

perentage_0 = 100 * nb_0 / total
perentage_1 = 100 * nb_1 / total

perentage_0
perentage_1

summary(data)
boxplot(data$V1, data$V2, data$V3)

# Correlation

library(corrplot)

M = cor(data_)
corrplot(M, method = "circle")

M0 = cor(data_0_)
corrplot(M0, method = "circle")

M1 = cor(data_1_)
corrplot(M1, method = "circle")

# K means
# Used with V5

clusters <- function(x, centers) {
  i = 1
  selected = 1
  d = abs(centers[1]-x)
  for (c in centers) {
    di = abs(c-x)
    if (di < d) {
      selected = i
    }
    i = i + 1
  }
  return(selected)
}

clusterst <- function(x, centers) {
  r = 1:length(x)
  i = 1
  for (xi in x) {
    r[i] = clusters(xi, centers)
    i = i + 1
  }
  return(r)
}

cl = kmeans(data$V5, 5)
cl$centers
for (i in c(1,2,3,4,5)) {
  c = clusters(data$V5[i], cl$centers)
  print(data$V5[i])
  print(c)
  print('')
}

tab = clusterst(data$V5, cl$centers)
tab = data.frame(tab)

data_extended = data
data_extended$cluster = tab
