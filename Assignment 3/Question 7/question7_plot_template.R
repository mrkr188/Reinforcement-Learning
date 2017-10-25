
# number of steps taken in each of 100 episodes, for values of n =1,5,10,10
n1 = c()

n5 = c()

n10 = c()

n20 = c()

# plot episodes vs number os steps
plot(x=n1, y=1:100, type="l", xlab="episodes", ylab="steps")
lines(x=n5, y=1:100, col = "red")
lines(x=n10, y=1:100, col = "blue")
lines(x=n20, y=1:100, col = "green")

legend(1,100, legend=c("n=1", "n=5", "n=10", "n=20"),
       col=c("black", "red", "blue", "green"), lty=1, cex=0.8)
legend("top", legend=c("alpha=0.5", "epsilon=0.1"))