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



void saveResults(double* data, int dataSize, const char* filename);

int main(int argc, char *argv[]) {
    srand(100);
    int k,i;
    int numEpisodes = 100;
    int numRuns = 100;
    // int total_steps;
    
    double result[numEpisodes];
    memset(result, 0, sizeof(result));
    
    // printf("\nPrinting number of runs every time: %d total Runs to complete\n",numRuns);
    for (k =0; k<numRuns; k++)
    {
        RL_init();

        // total_steps = 0;
        
        for (i = 0; i<numEpisodes; i++) {

            RL_episode(0);

            // total_steps += RL_num_steps();
            // result[i] += total_steps; 
        }

        RL_cleanup();
        // printf("%d\n",k);
        fflush( stdout );
    }
    
    printf("\n");
    
    /* average over runs */
    // for (i = 0; i < 100; i++){
    //     result[i] = result[i]/numRuns;
    //     printf("%d, ", result[i]);
    // }
    
    //save averaged number of steps taken at each episode
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
