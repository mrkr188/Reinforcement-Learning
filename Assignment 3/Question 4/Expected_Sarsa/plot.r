# Author: Matthew Schlegel
# Purpose: Just a simple plotting util to read in data from RL_EXP_OUT.dat

dat = read.table("num_steps.dat")

plot(x = dat[,1], y = 1:length(dat[,1]), type="l", ylab="steps", xlab="episode", col="purple")

plot(x=dat, y=1:100, type="l", xlab="episodes", ylab="steps")