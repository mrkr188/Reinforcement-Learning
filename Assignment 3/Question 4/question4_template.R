# number of steps taken in each of 100 episodes, for sarsa and expected sarsa agents
sarsa = c()

exp_sarsa = c()


# plot episodes vs number os steps
plot(x=sarsa, y=1:100, type="l", xlab="episodes", ylab="steps")
lines(x=exp_sarsa, y=1:100, col = "red")


legend(1,100, legend=c("sarsa", "expected sarsa"),
       col=c("black", "red"), lty=1, cex=0.8)
legend("top", legend=c("alpha=0.5", "epsilon=0.1"))