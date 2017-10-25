# number of steps taken in each of 50 episodes

#nstep
q = c(689, 459, 321, 277, 215, 162, 172, 167, 156, 222, 144, 67, 80, 80, 81, 76, 104, 67, 70, 69, 73, 58, 63, 68, 64, 66, 67, 64, 63, 48, 56, 63, 57, 59, 68, 74, 60, 69, 67, 81, 75, 66, 77, 71, 73, 54, 69, 73, 71, 52)

#dynaq
ps = c(913, 466, 309, 258, 214, 132, 130, 117, 92, 109, 83, 72, 81, 85, 52, 54, 58, 43, 34, 32, 32, 26, 21, 28, 19, 18, 19, 18, 18, 18, 17, 17, 17, 17, 20, 17, 17, 17, 18, 17, 17, 16, 16, 17, 17, 17, 17, 17, 17, 17)

# plot episodes vs number os steps
plot(ps, type="l", xlab="episodes", ylab="steps")
lines(q, col = "red")



legend("top", legend=c("Prioritized Sweeping", "N-Step-Sarsa n=4,alpha=0.05"),col=c("black", "red"), lty=1, cex=0.8)
