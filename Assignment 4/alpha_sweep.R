steps = c(420554, 519835, 435986, 604532, 1750690, 2194446, 2630388)

plot(steps, xlab="episodes", ylab="steps")
lines(steps, col = "red")

legend("topleft", legend=c("On X-axis", "1 - alpha=0.025", "2 - alpha=0.05", "3 - alpha=0.1", "4 - alpha=0.2", "5 - alpha=0.4", "6 - alpha=0.5", "7 - alpha=0.8"))