# number of steps taken in each of 100 episodes, for sarsa and expected sarsa agents
es0 = c(428, 1020, 1305, 1528, 1718, 2068, 2201, 2534, 2667, 2868, 3181, 3427, 3663, 3846, 4037, 4281, 4446, 4573, 4727, 4869, 4992, 5180, 5276, 5392, 5525, 5666, 5818, 6029, 6147, 6259, 6376, 6459, 6535, 6616, 6792, 6939, 7018, 7103, 7210, 7322, 7405, 7469, 7584, 7700, 7760, 7866, 7963, 8165, 8235, 8403)

# plot episodes vs number os steps
plot(x=es0, y=1:50, type="l", xlab="steps", ylab="episodes")

