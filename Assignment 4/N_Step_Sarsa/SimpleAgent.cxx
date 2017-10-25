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

#include <iostream>
#include <vector>

#include "rlglue/Agent_common.h" /* Required for RL-Glue */
#include "rlglue/RL_glue.h"

/*** IMPORTANT: 
 * Any private variables to the environment must be declared static
 * Otherwise they are default public global variables that can be 
 * accessed in other files 
 ***/

static gsl_vector* local_action;
static action_t* this_action;
static action_t* last_action;
static gsl_vector* last_observation;

static int numActions = 4;

static double discount = 0.95;

static double epsilon = 0.1;
static double alpha = 0.05;

static int total_steps;
static int n_eps;

int num_runs = 0;
int result[50];
int total_result[50];

// a double array to store q(s,a) for the grid
double q_sa[60][4];

// n step sarsa 
static int n_sarsa = 4; 

// to save states and actions of previous steps
static int *S_seq;
static int *A_seq;

/* -----------------------------------------------------------
utility functions 
--------------------------------------------------------------*/

// return greedy action, with random tie breaking
int returnGreedyAction(double q_sa[60][4], int stp1){
  double max_q_sa = -DBL_MAX;
  int atp_max;

  std::vector<int> index;

  for(int i = 0; i < numActions ; i++) {
        if (max_q_sa < q_sa[stp1][i]) {
            max_q_sa = q_sa[stp1][i];
            atp_max = i;
        }
  }
  // randomly breaks tie
  for(int i = 0; i < numActions ; i++) {
        if (max_q_sa == q_sa[stp1][i]) {
            index.push_back(i);
        }
  }
  
  atp_max = index[randInRange(index.size())];
  return atp_max;
}


void agent_init()
{
  //Allocate Memory
  local_action = gsl_vector_calloc(1);
  this_action = local_action;
  last_action = gsl_vector_calloc(1);
  last_observation = gsl_vector_calloc(1);

  // set initial q_sa values to 0
  for(int i=0; i<60; i++){
    for(int j=0; j<4; j++){
      q_sa[i][j] = 0;
    }
  }

  for(int i=0; i<50; i++){
      result[i] = 0;
  }

  if(num_runs==0){
    for(int i=0; i<50; i++){
        total_result[i] = 0;
    }
  }

  //use two array to store the sequence of S,A in an episode
  S_seq = (int *)malloc(sizeof(int)*10000);
  A_seq = (int *)malloc(sizeof(int)*10000);

  num_runs++;

  n_eps = 0;
  total_steps = 0;

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
    atp1 = returnGreedyAction(q_sa, stp1);
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



  // get current state agent is in
  int stp1 = (int)gsl_vector_get(this_observation,0);

  // atp1 stores action taken in current state
  int atp1;

  // select a such that q_sa is max at stp1 state. store it in max_q_sa
  // action corresponding to max_q_sa is stored in atp_max
  
  int atp_max = returnGreedyAction(q_sa, stp1);
  double max_q_sa = q_sa[stp1][atp_max];


  // select best action 1-epsilon probability, 
  // and uniform randomply choose other actions with epsilon probability
  if(rand_un() <= epsilon){
    atp1 = randInRange(numActions);
  }
  else{    
    atp1 = atp_max;
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
    // G = -1*(n_sarsa) + q_sa[stp1][atp1];

    // generalized version to calculate G
    for(int i=0; i<n_sarsa; i++){
      G += pow(discount, i)*reward;
    }
    G += pow(discount, n_sarsa)*q_sa[stp1][atp1];

    q_sa[stp_t][atp_t] += alpha*(G - q_sa[stp_t][atp_t]);
  }

  // update q_sa
  // q_sa[stp0][atp0] += alpha*(reward + discount*max_q_sa - q_sa[stp0][atp0]);

  
  gsl_vector_set(local_action,0,atp1);

  gsl_vector_memcpy(last_observation,this_observation); /*save observation, useful on the next step */
  gsl_vector_memcpy(last_action,local_action); /*save action, useful on the next step */
  
  return this_action;
}

void agent_end(double reward) {
  /* final learning update at end of episode */

  // result[n_eps] = total_steps;
  // n_eps++;

  // // get previous state and action that agent took to get to terminal state
  // int stp0 = (int)gsl_vector_get(last_observation,0);
  // int atp0 = (int)gsl_vector_get(last_action,0);

  // // update q_sa
  // q_sa[stp0][atp0] += alpha*(reward + discount*0 - q_sa[stp0][atp0]);

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
      for(int i=0; i<n_sarsa; i++){
        G += pow(discount, i)*reward;
      }
      G += pow(discount, n_sarsa)*q_sa[stp1][atp1];

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
      for(int i=0; i<n_sarsa; i++){
        G += pow(discount, i)*reward;
      }
      G += pow(discount, n_sarsa)*q_sa[stp1][atp1];

      // update q_sa
      q_sa[stp_t][atp_t] += alpha*(G - q_sa[stp_t][atp_t]);
    }
  }


}

void agent_cleanup() {
  /* clean up mememory */
  // for(int i=0; i<50; i++){
  //   total_result[i] += result[i];
  // }

  // if(num_runs==30){
  //   for(int i=0; i<50; i++){
  //     printf("%d, ", total_result[i]/50);
  //   }
  // }

  

  gsl_vector_free(local_action);
  gsl_vector_free(last_observation);
}

const char* agent_message(const char* inMessage) {
  /* might be useful to get information from the agent */
  if(strcmp(inMessage,"what is your name?")==0)
  return "my name is skeleton_agent!";
  
  /* else */
  return "I don't know how to respond to your message";
}
