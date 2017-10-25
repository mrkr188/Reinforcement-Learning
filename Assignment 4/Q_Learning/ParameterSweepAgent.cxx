/*
 * Author: Adam White, Matthew Schlegel
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 * Modified from: SimpleAgent.cxx
 * Modified By: Matthew Schlegel
 * Modified On: 1/3/17
 *
 *
 * agent does *no* learning, selects actions randomly from the set of legal actions, ignores observation/state
 *
 */

#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "rlglue/Agent_common.h" /* Required for RL-Glue */

//Include the values for the parameter sweep.
#include "ParameterSweep.h"

/*** IMPORTANT: 
 * Any private variables to the environment must be declared static
 * Otherwise they are default public global variables that can be 
 * accessed in other files 
 ***/

static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

static double alpha = 0;

//Shared alphaValues array.
// const double alphaValues[];
// const double alphaValues[] = {0.001,0.01,0.0197531,0.0296296,0.0444444,0.0666667,0.1,0.15,0.225,0.3375,0.50625};

static int numActions = 10;
static int numStates = 10;


void agent_init()
{
      local_action = gsl_vector_calloc(1);
      this_action = local_action;
      last_observation = gsl_vector_calloc(1);
}

const action_t *agent_start(const observation_t *this_observation) {

      int stp1 = (int)gsl_vector_get(this_observation,0); /* how you convert observation to a int, if state is tabular */
      int atp1=randInRange(numActions);
      gsl_vector_set(local_action,0,atp1);

      gsl_vector_memcpy(last_observation,this_observation);/*save observation, might be useful on the next step */

      return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {

      int stp1 = (int)gsl_vector_get(this_observation,0);
      int atp1=randInRange(numActions);

      /* might do some learning here */
    
      gsl_vector_set(local_action,0,atp1);
      gsl_vector_memcpy(last_observation,this_observation);
	
      return this_action;
}

void agent_end(double reward) {
      /* final learning update at end of episode */
}

void agent_cleanup() {
      /* clean up mememory */
      gsl_vector_free(local_action);
      gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
      /* might be useful to get information from the agent */

      //Read message and parse from Experiment.
      int recievedint = atoi(inMessage);

      //The recieved integer coresponds to the alpha vector
      printf("Agent Recieved Alpha Index: %d, With alpha Value %f\n", recievedint, alphaValues[recievedint]);
      //Set values.
      alpha = alphaValues[recievedint];

      return "I responded to your message by setting my alpha value.";
}
