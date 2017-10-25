/*
 * Author: Adam White, Matthew Schlegel
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 *
 * agent takes actions based on q(s,a). 
 *
 */

#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <time.h>


#include "rlglue/Agent_common.h" /* Required for RL-Glue */

/*** IMPORTANT: 
 * Any private variables to the environment must be declared static
 * Otherwise they are default public global variables that can be 
 * accessed in other files 
 ***/

static gsl_vector* local_action;
static action_t* this_action;
static gsl_vector* last_observation;

static int numActions = 1;
static int numStates = 99;

static int num_steps = 0;

// a matrix to save returns for each state, action pairs encountered
// s in rows, a in cols
// 1-99 states with index 1 to 99. 0 index is left alone. easy to keep track
// static gsl_matrix* returns = gsl_matrix_alloc(100,100);
static int returns[100][100];

// a matrix to store q(s,a) calculates from returns
// static gsl_matrix* q_sa = gsl_matrix_alloc(100,100);
static double q_sa[100][100];

// matrix storing counts, i.e, number of times an action is met in state
// this is to keep track of number of times a state, action pair occured in all episodes combined
// used to calculate Q(s,a) by averaging returns  
// static gsl_matrix* total_count_sa = gsl_matrix_alloc(100,100); 
static int total_count_sa[100][100];

// for each episode we keep track how many times a state, action pair has occured
// when the episode terminates with reward 1, we add all +1 to Q(s,a) for state, action pairs with count > 1 
// if episode terminates with 0 reward, we do nothing
// static gsl_matrix* episodic_count_sa = gsl_matrix_alloc(100,100);
static int episodic_count_sa[100][100];

// vector storing policy
// indexes 1-99 store policy for states 1-99. 
static int policy[100]; 

// vector storing v(pi)
static double v_pi[100];

// averaging v_pi over multiple steps
static double v_pi_average[100];

// a function to return min of 2 arguments
inline int min ( int a, int b ) { return a < b ? a : b; }

// a function for saving results - copied from experiment file
void saveResults(double* data, int dataSize, const char* filename) {
  FILE *dataFile;
  int i;
  dataFile = fopen(filename, "w");
  for(i = 0; i < dataSize; i++){
    fprintf(dataFile, "%lf\n", data[i]);
  }
  fclose(dataFile);
}
// same save results function for int
void saveResults(int* data, int dataSize, const char* filename) {
  FILE *dataFile;
  int i;
  dataFile = fopen(filename, "w");
  for(i = 0; i < dataSize; i++){
    fprintf(dataFile, "%d\n", data[i]);
  }
  fclose(dataFile);
}

void agent_init()
{
  //Allocate Memory
  local_action = gsl_vector_calloc(1);
  this_action = local_action;
  last_observation = gsl_vector_calloc(1);

  // set returns to 0
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++)
      returns[i][j] = 0;

  // set q_sa to 0
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++)
      q_sa[i][j] = 0;

  // set total_count_sa to 0
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++)
      total_count_sa[i][j] = 0;
  
  // first episode that ever runs will have a random action uniformly selected for each state
  // monte carlo exploring starts
  for(int i=1; i<100; i++){
    // for a given state we have actions ranging from 1 to min(state,100-state)
    int num_actions = min(i, 100-i);
    policy[i] = randInRange(num_actions) + 1;

    // make v_pi 0 after each step is completed
    v_pi[i] = 0;
  }

}

const action_t *agent_start(const observation_t *this_observation) {
  
  // state
  int stp1 = (int)gsl_vector_get(this_observation,0); 

  // choose action such that all actions have same probability
  int atp1 = randInRange( min(stp1, 100-stp1) ) + 1;
  gsl_vector_set(local_action,0,atp1);
  
  gsl_vector_memcpy(last_observation,this_observation);/*save observation, might be useful on the next step */

  // set episodic_count_sa to 0 at the start of each episode
  for (int i = 0; i < 100; i++)
    for (int j = 0; j < 100; j++)
      episodic_count_sa[i][j] = 0;
  
  return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {
  
  int stp1 = (int)gsl_vector_get(this_observation,0);
  
  // at a given state stp1, get the action from policy vector
  int atp1 = policy[stp1];
  
  // increase count of state, action pairs, for this episode
  episodic_count_sa[stp1][ atp1]++;
  
  gsl_vector_set(local_action,0,atp1);
  gsl_vector_memcpy(last_observation,this_observation);
  
  return this_action;
}

void agent_end(double reward) {
  // update all stuff at the end of episode

  // loop over all action state pairs
  // i = state, j = action
  for (int i = 1; i < 100; i++){
    for (int j = 1; j < 100; j++){
      if(episodic_count_sa[i][j] > 0){

        // update total count of state, action pairs over all episodes combined
        total_count_sa[i][j] += episodic_count_sa[i][j];
        
        // update returns
        returns[i][j]++;

        // update q_sa
        // there is no worries of getting total counts zero in the denominator, since we have if condition
        // which checks if the episodic count > 0
        q_sa[i][j] = (double)returns[i][j]/total_count_sa[i][j];
      }
    }
  }

  // update policy with greedy action 
  // at the end of this loop we return action with maximum q(s,a) for a given state
  for(int state=1; state<100; state++){

    int num_actions = min(state, 100-state); // number of actions for a state

    // if all of the q(s,a) for a state are 0, then return an action selected random uniformly - monte carlo es
    // select action random uniformly
    double optimal_action = randInRange( num_actions ) + 1;
    double max_q_sa = 0;
    for(int action=1; action<=num_actions; action++){
      if(returns[state][action] > max_q_sa){
        max_q_sa = q_sa[state][action];
        optimal_action = action;
      }
    }
    // return best policy
    policy[state] = optimal_action;
    v_pi[state] = max_q_sa;
    }

  }





void agent_cleanup() {
  /* clean up memory */
  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);

  num_steps++;

  // these 3 for loops are to calculate average of v_pi over all the steps
  if(num_steps == 1){
    for(int i=1; i<100; i++){
      v_pi_average[i] = v_pi[i];
    }
  }

  for(int i=1; i<100; i++){
      v_pi_average[i] += v_pi[i];
  }

  if(num_steps == 30){
    for(int i=1; i<100; i++){
      v_pi_average[i] = v_pi_average[i]/30;
    }
  }

  // after all episodes are completed save v(pi) and policy to a file
  saveResults(v_pi_average, 100, "v_pi.dat");
  saveResults(policy, 100, "policy.dat");  

}

const char* agent_message(const char* inMessage) {
  /* might be useful to get information from the agent */
  if(strcmp(inMessage,"what is your name?")==0)
  return "my name is skeleton_agent!";
  
  /* else */
  return "I don't know how to respond to your message";
}
