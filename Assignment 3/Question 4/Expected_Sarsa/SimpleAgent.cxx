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



#include "rlglue/Agent_common.h" /* Required for RL-Glue */

/*** IMPORTANT: 
 * Any private variables to the environment must be declared static
 * Otherwise they are default public global variables that can be 
 * accessed in other files 
 ***/

static gsl_vector* local_action;
static action_t* this_action;
static action_t* last_action;
static gsl_vector* last_observation;

static int numActions = 8;

static double discount = 1.0;

static double epsilon = 0.0001;
static double alpha = 0.5;

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
  int max_a; // used to save action with maximum q_sa
  // rind action with maximum q_sa
  double temp = -DBL_MAX ;
  for (int i = 0; i < numActions ; i++) {
      if (temp <= q_sa[stp1][i]) {
          temp = q_sa[stp1][i];
          max_a = i;
      }
  }

  // select best action 1-epsilon probability, 
  // and uniform randomply choose other actions with epsilon probability
  if(rand_un() <= epsilon){
    atp1 = randInRange(numActions);
  }
  else{
    atp1 = max_a;
  }

  // printf("stp1, atp1 - %d, %d\n", stp1, atp1);
  // scanf ("%d",&dummy);

  // find E[Q(stp1,atp1)|stp1]
  double est_q_sa = 0;
  for(int i=0; i<8; i++){
    if(atp1 == max_a){
      est_q_sa += q_sa[stp1][atp1]*(1 - epsilon);
    }
    else{
      est_q_sa += q_sa[stp1][atp1]*(epsilon/7);
    }
  }

  // update q_sa with the estimated E[Q(stp1,atp1)|stp1]
  q_sa[stp0][atp0] += alpha*(-1 + discount*est_q_sa - q_sa[stp0][atp0]);
  
  // if(n_steps % 1000 == 0){
  //   printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
  //   int dummy;
  //   scanf ("%d",&dummy);
  // }

  gsl_vector_set(local_action,0,atp1);

  gsl_vector_memcpy(last_observation,this_observation); /*save observation, useful on the next step */
  gsl_vector_memcpy(last_action,local_action); /*save action, useful on the next step */
  
  return this_action;
}

void agent_end(double reward) {
  /* final learning update at end of episode */
  // printf("episode %d - %d steps\n", n_eps, n_steps);
  result[n_eps] = total_steps;
  n_eps++;
}

void agent_cleanup() {
  /* clean up mememory */
  for(int i=0; i<100; i++){
    total_result[i] += result[i];
  }

  if(num_runs==100){
    for(int i=0; i<100; i++){
      printf("%d, ", total_result[i]/100);
    }
  }

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
