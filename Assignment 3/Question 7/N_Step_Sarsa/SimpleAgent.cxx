/*
 * Author: Adam White, Matthew Schlegel
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 *
 * agent does *no* learning, selects actions randomly from the set of legal actions, ignores observation/state
 *
 */

#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>



#include "rlglue/Agent_common.h" /* Required for RL-Glue */
#include "rlglue/RL_glue.h"

/*** IMPORTANT: 
 * Any private variables to the environment must be declared static
 * Otherwise they are default public global variables that can be 
 * accessed in other files 
 ***/

 #define REWARD -1

static gsl_vector* local_action;
static action_t* this_action;
static action_t* last_action;
static gsl_vector* last_observation;

static int numActions = 8;

static double discount = 1.0;

static double epsilon = 0.1;
static double alpha = 0.5;

// n step sarsa 
static int n_sarsa = 5; 

// to save states and actions of previous steps
static int *S_seq;
static int *A_seq;

// to save number of steps
// somehow not able to make RL_num_steps() work, ***have to look into it***

static int total_steps;
static int n_eps;

int num_runs = 0;
int result[100];
int total_result[100];

// a double array to store q(s,a) for the grid
double q_sa[70][8];

void agent_init()
{
  //Allocate Memory
  local_action = gsl_vector_calloc(1);
  this_action = local_action;
  last_action = gsl_vector_calloc(1);
  last_observation = gsl_vector_calloc(1);

  // set initial q_sa values to 0
  for(int i=0; i<70; i++){
    for(int j=0; j<8; j++){
      q_sa[i][j] = 0;
    }
  }

  for(int i=0; i<100; i++){
      result[i] = 0;
  }

  if(num_runs==0){
    for(int i=0; i<100; i++){
        total_result[i] = 0;
    }
  }

  num_runs++;

  n_eps = 0;
  total_steps = 0;

  //use two array to store the sequence of S,A in an episode
  S_seq = (int *)malloc(sizeof(int)*10000);
  A_seq = (int *)malloc(sizeof(int)*10000);

}

const action_t *agent_start(const observation_t *this_observation) {

  int stp1 = (int)gsl_vector_get(this_observation,0); /* how you convert observation to a int, if state is tabular */

  int atp1;

  // select best action with 1-epsilon probability, 
  // and uniform randomply choose other actions with epsilon probability
  if(rand_un() <= epsilon){
    atp1 = randInRange(numActions);
  }
  else{
    double temp = -DBL_MAX;
    for (int i = 0; i < numActions ; i++) {
        if (temp <= q_sa[stp1][i]) {
            temp = q_sa[stp1][i];
            atp1 = i;
        }
    }
  }

  //store current state and action into sequence
  S_seq[0] = stp1;
  A_seq[0] = atp1;

  gsl_vector_set(local_action,0,atp1);
  
  gsl_vector_memcpy(last_observation,this_observation); /*save observation, useful on the next step */
  gsl_vector_memcpy(last_action,local_action); /*save action, useful on the next step */
  
  return this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {

  total_steps++;
  
  // get previous state and action that agent took to get to current state
  int stp0 = (int)gsl_vector_get(last_observation,0);
  int atp0 = (int)gsl_vector_get(last_action,0);

  // printf("stp0, atp0 - %d, %d\n", stp0, atp0);
  // int dummy;
  // scanf ("%d",&dummy);

  // get current state agent is in
  int stp1 = (int)gsl_vector_get(this_observation,0);

  // atp1 stores action taken in current state
  int atp1;

  // select best action 1-epsilon probability, 
  // and uniform randomply choose other actions with epsilon probability
  if(rand_un() <= epsilon){
    atp1 = randInRange(numActions);
  }
  else{
    double temp = -DBL_MAX ;
    for (int i = 0; i < numActions ; i++) {
        if (temp <= q_sa[stp1][i]) {
            temp = q_sa[stp1][i];
            atp1 = i;
        }
    }
  }

  // current step number
  int n = RL_num_steps()-1;

  // store current state and action into sequence
  S_seq[n] = stp1;
  A_seq[n] = atp1;

  // update q_sa
  if(n > n_sarsa-1){

    int stp_t = S_seq[n-n_sarsa];
    int atp_t = A_seq[n-n_sarsa];

    // calculate G
    double G = 0;

    // hacky version to calculate G
    G = -1*(n_sarsa) + q_sa[stp1][atp1];

    // generalized version to calculate G
    // for(int i=0; i<n_sarsa; i++){
    //   G += pow(discount, i)*REWARD;
    // }
    // G += pow(discount, n_sarsa)*q_sa[stp1][atp1];

    q_sa[stp_t][atp_t] += alpha*(G - q_sa[stp_t][atp_t]);
  }


  gsl_vector_set(local_action,0,atp1);

  gsl_vector_memcpy(last_observation,this_observation); /*save observation, useful on the next step */
  gsl_vector_memcpy(last_action,local_action); /*save action, useful on the next step */
  
  return this_action;
}

void agent_end(double reward) {

  /* final learning update at end of episode */

  result[n_eps] = total_steps;
  n_eps++;

  int n = RL_num_steps()-1;

  // do updates for remaining state, action pairs
  // n_sarsa-1 state, action pairs are left at the end of the arrays S_seq and A_seq, which needs to be upgraded

  int stp_t, atp_t, stp1, atp1;

  // stores action, state pair just bfore reaching terminal state
  stp1 = S_seq[n];
  atp1 = A_seq[n];

  double G;

  // if number of steps in episode are greater than n_sarsa
  if(n > n_sarsa-1){
    for(int i=0; i < n_sarsa-1; i++){
      
      stp_t = S_seq[n-n_sarsa+1+i];
      atp_t = A_seq[n-n_sarsa+1+i];

      G = 0;
      // calculate G
      G = -1*(n_sarsa-1) + q_sa[stp1][atp1];

      // update q_sa
      q_sa[stp_t][atp_t] += alpha*(G - q_sa[stp_t][atp_t]);
    }
  }
  // if the number of steps in episode are smaller then the n_sarsa
  else {
        for(int i=0; i < n; i++){
      
      stp_t = S_seq[i];
      atp_t = A_seq[i];

      G = 0;
      // calculate G
      G = -1*(n_sarsa-1) + q_sa[stp1][atp1];

      // update q_sa
      q_sa[stp_t][atp_t] += alpha*(G - q_sa[stp_t][atp_t]);
    }
  }

}

void agent_cleanup() {
  /* clean up mememory */
  for(int i=0; i<100; i++){
    total_result[i] += result[i];
  }

  // at the end of all runs(here its 100), print number of steps taken for each episode onto the console
  if(num_runs==100){
    for(int i=0; i<100; i++){
      printf("%d, ", total_result[i]/100);
    }
  }

  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
  gsl_vector_free(last_action);
}

const char* agent_message(const char* inMessage) {
  /* might be useful to get information from the agent */
  if(strcmp(inMessage,"what is your name?")==0)
  return "my name is skeleton_agent!";
  
  /* else */
  return "I don't know how to respond to your message";
}
