# number of steps taken in each of 50 episodes

#q
q = c(824, 620, 455, 297, 232, 285, 166, 130, 165, 131, 151, 127, 91, 91, 75, 73, 66, 47, 49, 48, 50, 31, 28, 27, 29, 22, 22, 19, 19, 18, 17, 17, 19, 18, 20, 18, 18, 18, 18, 17, 17, 17, 17, 17, 18, 18, 17, 17, 17, 18)

#dynaq
ps = c(913, 466, 309, 258, 214, 132, 130, 117, 92, 109, 83, 72, 81, 85, 52, 54, 58, 43, 34, 32, 32, 26, 21, 28, 19, 18, 19, 18, 18, 18, 17, 17, 17, 17, 20, 17, 17, 17, 18, 17, 17, 16, 16, 17, 17, 17, 17, 17, 17, 17)

# plot episodes vs number os steps
plot(ps, type="l", xlab="episodes", ylab="steps")
lines(q, col = "red")



legend("top", legend=c("Prioritized Sweeping", "Q-Learning"),col=c("black", "red"), lty=1, cex=0.8)
