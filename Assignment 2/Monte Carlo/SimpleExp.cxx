/*
 * Author: Adam White, Matthew Schlegel
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 * Last Modified by: Matthew Schlegel
 * Last Modified on: 1/6/2017
 *
 * Experiment runs 100, 10000, 10000000 episodes
 * Monte Carlo ES applied to gambler's problem from 4th chapter
 * 
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "rlglue/RL_glue.h" /* Required for RL-Glue  */

int main(int argc, char *argv[]) {

  int numEpisodes = 1000000;
  int maxStepsInEpisodes = 100;
  int numSteps = 30;
  
  // run steps
  for (int k =0; k<numSteps; k++)
  {
    RL_init();
    // run episodes
    for (int i =0; i<numEpisodes; i++) {
      RL_episode(maxStepsInEpisodes);
    }
    RL_cleanup();
    printf("................................%d \n", k);
    fflush( stdout );
  }

  RL_init();
  

  
  
  printf("\nDone\n");
    
  return 0;
}

