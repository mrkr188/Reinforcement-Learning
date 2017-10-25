v_pi = read.table("v_pi.dat")
policy = read.table("policy.dat")

par(mfrow=c(2,1))
plot(x = 1:length(v_pi[,1]), y = v_pi[,1], type="l", ylab="v(pi)", xlab="state")
plot(x = 1:length(policy[,1]), y = policy[,1], type="l", ylab="policy", xlab="state")

