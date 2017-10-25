/*
 * Author: Adam White, Matthew Schlegel
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 * Last Modified by: Matthew Schlegel
 * Last Modified on: 1/6/2017
 *
 * experiment runs 200 episodes, averaging the cummulative reward per episode over 30 independent runs.
 * Results are saved to file.
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "rlglue/RL_glue.h" /* Required for RL-Glue  */
#include <iostream>


void saveResults(double* data, int dataSize, const char* filename);

int main(int argc, char *argv[]) {
    srand(100);
    int k,i;
    int numEpisodes = 50;
    int numRuns = 30;
    
    int result[numEpisodes];
    memset(result, 0, sizeof(result));
    
    // printf("\nPrinting number of runs every time: %d total Runs to complete\n",numRuns);
    for (k =0; k<numRuns; k++)
    {
        RL_init();
        
        for (i = 0; i<numEpisodes; i++) {
            RL_episode(2000);
            int steps = RL_num_steps();
            // printf("run - %d, episode - %d, steps - %d\n", k+1, i+1, steps);
            result[i] = result[i] + steps;
        }

        RL_cleanup();
        
        fflush( stdout );
    }
    printf("\nDone\n");
    
    /* average over runs */
    for (i = 0; i < numEpisodes; i++){
        result[i] = result[i]/numRuns;
        printf(" %d,", result[i]);
    }
    printf("\n");
    
    // save averaged number of steps taken at each episode
    // saveResults(result, numEpisodes+1, "num_steps.dat"); 
    
    return 0;
}

void saveResults(double* data, int dataSize, const char* filename) {
  FILE *dataFile;
  int i;
  dataFile = fopen(filename, "w");
  for(i = 0; i < dataSize; i++){
    fprintf(dataFile, "%lf\n", data[i]);
  }
  fclose(dataFile);
}
