rm(list=ls())

pckgs <- c("RODBC", "dplyr","lda","LDAvis","RWeka", "tm")

lapply(pckgs, require, character.only = TRUE)


raw <- read.csv("busNeeds.csv")


# combine fingerprint text in one field and filter those with less than 3 characters. 


resp <- (paste(raw$Answer_Text, sep=" "))

#names(resp) <- fpd_combined$opid


resp <- gsub("[[:punct:]]"," ",resp)
resp <- gsub("[[:digit:]]","",resp)
resp <- tolower(resp)
stop_words <- c(stopwords("SMART"),"hp")
resp <- unlist(lapply(resp, function(x) {
    rtn <- setdiff(unlist(strsplit(x," ")),stop_words) 
    paste(rtn[rtn!=""], collapse=" ")
    
}))

doc.list <- lapply(resp, NGramTokenizer, control=Weka_control(max=2, min=1))



# metagtmc <- fpd_raw %>% inner_join(fpd_combined, by = c("opid"="opid")) %>% group_by(opid) %>% summarise(gtmc_vec = paste(gtmc,collapse=","))
# 
# 
# 


#tokenize on space and output as a list:



# 
# # compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)


# 
# # remove terms that are stop words or occur fewer than 5 times:
del <-  term.table < 2 
term.table <- term.table[!del]
vocab <- names(term.table)
# 
# # now put the documents into the format required by the lda package:
get.terms <- function(x) {
    index <- match(x, vocab)
    index <- index[!is.na(index)]
    rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)
# 
D <- length(documents)  # number of documents (594)
W <- length(vocab)  # number of terms in the vocab (589)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document [312, 288, 170, 436, 291, ...]
N <- sum(doc.length)  # total number of tokens in the data (546,827)
term.frequency <- as.integer(term.table) 

K <- 20
G <- 5000
alpha <- 0.02
eta <- 0.02

# Fit the model:
library(lda)
set.seed(357)
t1 <- Sys.time()
fit <- lda.collapsed.gibbs.sampler(documents = documents, K = K, vocab = vocab, 
                                   num.iterations = G, alpha = alpha, 
                                   eta = eta, initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)
t2 <- Sys.time()
t2 - t1
theta <- t(apply(fit$document_sums + alpha, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + eta, 2, function(x) x/sum(x)))

fingerprints <- list(phi = phi,
                     theta = theta,
                     doc.length = doc.length,
                     vocab = vocab,
                     term.frequency = term.frequency)
library(gistr)
gist_auth()
json <- createJSON(phi = fingerprints$phi, 
                   theta = fingerprints$theta, 
                   doc.length = fingerprints$doc.length, 
                   vocab = fingerprints$vocab, 
                   term.frequency = fingerprints$term.frequency)

serVis(json, as.gist=TRUE)
