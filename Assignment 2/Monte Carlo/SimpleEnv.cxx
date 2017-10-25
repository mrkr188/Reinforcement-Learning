/*
 * Author: Adam White
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 *
 * env is simulation of gambler's problem from chapter 4. 
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "rlglue/Environment_common.h" /* Required for RL-Glue interface*/
#include "utils.h"

/*** IMPORTANT: 
 * Any private variables to the environment must be declared static
 * Otherwise they are default public global variables that can be 
 * accessed in other files 
 ***/

static gsl_vector* local_observation;
static reward_observation_terminal_t this_reward_observation;
static const int nStatesSimpleEnv = 101;

// probability of heads
static double p_heads = 0.25;

void env_init() 
{
	local_observation = gsl_vector_calloc(1);
	
	this_reward_observation.observation=local_observation;
	this_reward_observation.reward=0;
	this_reward_observation.terminal=0;
	
	return;
}

const observation_t* env_start()
{
  // choose a initial state such that all states 1-99 have equal probability
  gsl_vector_set(local_observation, 0, randInRange(99)+1);
  return this_reward_observation.observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
  // for terminal state make this 1. helps in terminating episode
  int episode_over=0;

  // the reward = 1 if (state + action) >= 100, 0 if (state - action) = -1
  // reward = 0  for all other states
  double the_reward = 0;
  
  int atp1 = gsl_vector_get(this_action,0); /* how to extact action */

  // to store next step.
  int stp1;

  // with p_heads probability the state transition happens to (state + action)
  // else transition happens to (state - action)
  if(rand_un() < p_heads){
    stp1 = gsl_vector_get(local_observation,0) + atp1; 
  }else{
    stp1 = gsl_vector_get(local_observation,0) - atp1;
  }

  // if terminal state 100 is reached reward 1. if terminal state 0 is reached reward 0
  if(stp1 >= 100){
    // if (state + action) exceeds 100 make it 100
    stp1 = 100;
    episode_over = 1; // terminate episode 
    the_reward = 1;
    }
    else if(stp1 == 0){
      episode_over = 1;
      the_reward = -1;
    }
  
  gsl_vector_set(local_observation, 0,stp1);
  this_reward_observation.reward = the_reward;
  this_reward_observation.terminal = episode_over;
  
  return &this_reward_observation;
}

void env_cleanup()
{
  gsl_vector_free(local_observation);
}

const char* env_message(const char* inMessage) 
{
  if(strcmp(inMessage,"what is your name?")==0)
  return "my name is skeleton_environment!";
  
  /* else */
  return "I don't know how to respond to your message";
}
