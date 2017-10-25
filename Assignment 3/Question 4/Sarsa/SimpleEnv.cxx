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

7x10 gird starts at 00 which is (0,0), row 0 and column 0
and ends at 69 which is (6,9)

s/10 gives row and s%10 gives column
basically just put a ',' between 2 digits and that gives rows and columns

here is the grid layout
00  01  02  03  04  05  06  07  08  09(0,9)
10  11  12  13  14  15  16  17  18  19
20  21  22  23  24  25  26  27  28  29
30  31  32  33  34  35  36  37  38  39(3,9)
40  41  42  43  44  45  46  47  48  49
50  51  52  53  54  55  56  57  58  59
60  61  62  63  64  65  66  67  68  69(6,9)

30 is start
37 is goal


actions representation

consider north = up, east = right, south = down, west = left
0 is north west, 1 is north(up), 2 is north east 
3 is left, 4 is right
5 is south west, 6 is south, 7 is south east
   0  1  2
   3     4
   5  6  7
   
*/

// wind direction at each column
static int wind[10] = {0,0,0,1,1,1,2,2,1,0};



// given current state and action it gives next state
// note that state transition is deterministic
int stateTransition(int old_state, int action){
   
   int new_state, row, col;
   
   row = old_state/10;
   col = old_state%10;
   
   // NORTH  WEST
   if(action == 0){
      row = max(row - 1 - wind[col], 0);
      col = max(col - 1, 0);
   }
   // NORTH 
   else if(action == 1){
      row = max(row - 1 - wind[col], 0);
      // col = col;
   }
   // NORTH EAST
   else if(action == 2){
      row = max(row - 1 - wind[col], 0);
      col = min(col + 1, 9);
   }
   // WEST
   else if(action == 3){
      // row = row;
      col = max(col - 1, 0);
   }
   // EAST
   else if(action == 4){
      // row = row;
      col = min(col + 1, 9);
   }
   // SOUTH WEST
   else if(action == 5){
      row = max( min(row + 1 - wind[col], 6), 0);
      col = max(col - 1, 0);
   }
   // SOUTH
   else if(action == 6){
      row = max( min(row + 1 - wind[col], 6), 0);
      // col = col;
   }
   // SOUTH EAST
   else{ // action == 7
      row = max( min(row + 1 - wind[col], 6), 0);
      col = min(col + 1, 9);
   }
   
   new_state = row*10 + col;
   
   return new_state;
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
  // set initial state to 30
  gsl_vector_set(local_observation,0,30);

  return this_reward_observation.observation;
}

const reward_observation_terminal_t *env_step(const action_t *this_action)
{
  // get previous state from local observation
  int stp1 = (int)gsl_vector_get(local_observation,0);

  // terminates episode when this is set to 1
  int episode_over = 0;
  // reward is -1 for each step until terminal state
  double the_reward = -1.0;

  // get action from agent
  int atp1 = gsl_vector_get(this_action,0);
  
  // state transitions are deterministic governed by principles implemented in stateTransition function above
  // update the new state after state transition happens
  stp1 = stateTransition(stp1, atp1); 

  // terminate episode when agent reaches goal at 37 
  if(stp1 == 37)
    episode_over = 1; 
  
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
