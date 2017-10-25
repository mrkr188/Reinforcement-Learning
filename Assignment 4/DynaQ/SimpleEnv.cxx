/*
 * Author: Adam White
 * Purpose: for use of Rienforcement learning course Indiana University Spring 2016
 *
 * env transitions *ignore* actions, state transitions, rewards, and terminations are all random
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
static const int nStatesSimpleEnv = 20;


// a function to return min and max of 2 arguments
inline int min ( int a, int b ) { return a < b ? a : b; }
inline int max ( int a, int b ) { return a > b ? a : b; }


/* 
grid world representation

6x9 gird starts at 00 which is (0,0), row 0 and column 0
and ends at 69 which is (6,9)

s/10 gives row and s%10 gives column
basically just put a ',' between 2 digits and that gives rows and columns
the states in [], such as [11] are blocks

here is the grid layout
   00  01   02   03  04   05   06  [07]  08{G}
   10  11  [12]  13  14   15   16  [17]  18
{S}20  21  [22]  23  24   25   26  [27]  28
   30  31  [32]  33  34   35   36   37   38 (3,8)
   40  41   42   43  44  [45]  46   47   48
   50  51   52   53  54   55   56   57   58 (5,8)
 (5,0)    (5,2)

20 is start
08 is goal


actions representation

0 up
1 left
2 right
3 down

      0   
   1     2
      3   
   
*/


// given current state and action it gives next state
// note that state transition is deterministic
int stateTransition(int old_state, int action){
   
   int new_state, row, col;
   
   row = old_state/10;
   col = old_state%10;
   
   // UP
   if(action == 0){
      row = max(row - 1, 0);
      // col = col;
   }
   // LEFT 
   else if(action == 1){
      // row = row;
      col = max(col - 1, 0);
   }
   // RIGHT
   else if(action == 2){
      // row = row;
      col = min(col + 1, 8);
   }
   // DOWN
   else //action == 3
   {
      row = max( min(row + 1, 5), 0);
      // col = col;
   }

   
   new_state = row*10 + col;

   if(new_state == 12 || new_state == 22 || new_state == 32 || new_state == 45 || new_state == 07 || new_state == 17 || new_state == 27){
     return old_state;
   }else{
     return new_state;}
}


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
  // set initial state to 20
  gsl_vector_set(local_observation,0,20);

  return this_reward_observation.observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
  // get previous state from local observation
  int stp1 = (int)gsl_vector_get(local_observation,0);

  // terminates episode when this is set to 1
  int episode_over = 0;

  // reward is 0 for each step until terminal state
  double the_reward = 0;

  // get action from agent
  int atp1 = gsl_vector_get(this_action,0);
  
  // state transitions are deterministic governed by principles implemented in stateTransition function above
  // update the new state after state transition happens
  stp1 = stateTransition(stp1, atp1); 

  // terminate episode when agent reaches goal at 08 
  if(stp1 == 8){
    episode_over = 1; 
    the_reward = 1.0;
  }

  // printf("%f ", the_reward);
  
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
