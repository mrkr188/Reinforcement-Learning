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
#include <iostream>

#include "rlglue/RL_glue.h" /* Required for RL-Glue  */

//Include the values for the parameter sweep.
#include "ParameterSweep.h"


void saveResults(double* data, int dataSize, const char* filename);

int main(int argc, char *argv[]) {
    srand(100);
    int k,i,alpha_ind;
    int numEpisodes = 50;
    int numRuns = 30;
    
    int result[7];
    memset(result, 0, sizeof(result));


    for(alpha_ind = 0; alpha_ind < alphaValuesSize; alpha_ind++){

        //Get the alpha value.
        printf("Experiment Alpha Value: %f\n", alphaValues[alpha_ind]);
        //Make a string
        char agent_message[10];
        //Print to string what you want to send. In this case the index coresponding to the alpha value desired.
        sprintf(agent_message, "%d", alpha_ind);
        //Send message.
        RL_agent_message(agent_message);
        
        printf("\nPrinting one dot for every run: %d total Runs to complete\n",numRuns);

        int tot_steps = 0;

        for (k =0; k<numRuns; k++){

            RL_init();

            for (i =0; i<numEpisodes; i++) {
                RL_episode(2000);
                tot_steps += RL_num_steps();
            }
            RL_cleanup();
            printf(".");
            fflush( stdout );
        }
        printf("\n");
        result[alpha_ind] = tot_steps;
    }
    
    printf("\nDone\n");
    
    /* average over runs */
    for (i = 0; i < 7; i++){
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
