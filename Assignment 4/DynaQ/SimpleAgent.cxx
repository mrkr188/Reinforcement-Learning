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
#include <stdlib.h>     /* for atoi */
#include <math.h>       /* for fabs */

#include <iostream>
#include <map>
#include <string>
#include <sstream>
#include <queue>
#include <set>
#include <iterator>



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

static int numActions = 4;

static double discount = 0.95;

static double epsilon = 0.1;
static double alpha = 0.1;

static double theta = 0.5;

static double psteps = 5;

static int total_steps;
static int n_eps;

int num_runs = 0;
int result[50];
int total_result[50];

// a double array to store q(s,a) for the grid
double q_sa[60][4];

// a map to store model
static std::map< std::string, std::vector<int> > model;

// a map storing all S,A leading to particular state
static std::map< int, std::set<std::string> > previous_sa;

/* -----------------------------------------------------------
Queue 
--------------------------------------------------------------*/

class saPair{
public:
    std::string sa;
    double priority;
    saPair(std::string x, double y){
        sa = x;
        priority = y;
    }
};

bool operator< (const saPair& a, const saPair& b) {
    return a.priority < b.priority;
}

// a priority queue storing s,a pairs with priority 
static std::priority_queue<saPair> pqueue;

// a map which contains S,A pairs inserted into the pqueue
// we cant avoid inserting already present S,A pair into the STL priority_queue
// so to avoid duplicate S,A pairs, I maintain a map
static std::map<std::string, double> prioritymap;

// insert S,A pair into the prioritymap
// if S,A pair is already present in the map update priority, if the new priority is more than the one in the map
void prioritymapInsert(std::map<std::string, double> &pmap, std::string s, double value){
    std::map<std::string, double>::iterator itr = pmap.find(s);
    if (itr == pmap.end()){
        pmap[s] = value;
    }
    else if(pmap[s] < value){
        pmap[s] = value;
    }else{
        
    }
}


/* -----------------------------------------------------------
utility functions 
--------------------------------------------------------------*/

// print map. to print model and view what states are visited
void printMap(std::map< std::string, std::vector<int> > &m){
    for(std::map< std::string, std::vector<int> >::iterator itr = m.begin(); itr != m.end(); ++itr){
	    std::cout<< itr->first << " " << itr->second[0] << "," << itr->second[1] << std::endl;
	}
}

// takes S,A and return string with action first and state next
// for example given 12 state and 1 action, this returns "112" string
std::string makeSAstring(int s, int a){
  std::ostringstream ss;
  ss << a << s; 
  return ss.str();
}

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


  num_runs++;
  std::cout << num_runs << std::endl;

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
  double max_q_sa;
  int atp_max;

  
  atp_max = returnGreedyAction(q_sa, stp1);
  max_q_sa = q_sa[stp1][atp_max]; 

  // select best action 1-epsilon probability, 
  // and uniform randomply choose other actions with epsilon probability
  if(rand_un() <= epsilon){
    atp1 = randInRange(numActions);
  }
  else{
    atp1 = atp_max;
  }


  // a string containing S,A - this is key of model map      
  std::string sa0 = makeSAstring(stp0, atp0);

  if (model.find(sa0) == model.end()){

    // a vector containing next state and reward - this is value of model map
    std::vector<int> next_sr;
    next_sr.push_back(stp1);
    next_sr.push_back(reward);

    // insert S,A into model map
    model[sa0] = next_sr;
  }

  
  // insert the previous S,A into previous_sa map. 
  // key for the previous_sa map is just the current state
  // value of previous_sa map is set containing all previous S,A pairs

  if(previous_sa.find(stp1) == previous_sa.end()){
    
    std::set<std::string> prev_sa;
    prev_sa.insert(sa0);

    previous_sa[stp1] = prev_sa;
  } 
  else{
    previous_sa[stp0].insert(sa0);
  }


  // update q_sa
  // q_sa[stp0][atp0] += alpha*(reward + discount*max_q_sa - q_sa[stp0][atp0]);
  // printf("%f--- ", q_sa[stp0][atp0]);

  double P = fabs(reward + discount*max_q_sa - q_sa[stp0][atp0]);

  // printf("%f--- ", P);


  
  if(P > theta){
    // insert into PQueue
    pqueue.push(saPair(sa0, P));

    // insert into prioritymap
    prioritymapInsert(prioritymap, sa0, P);
  }

  q_sa[stp0][atp0] += alpha*(reward + discount*max_q_sa - q_sa[stp0][atp0]);

  int planningStep = 0;

  while(planningStep < psteps && !pqueue.empty()){

    planningStep++;

    // take out the first element of PQueue
    saPair first = pqueue.top();

    // this loop ensures that the S,A pairs that are inserted multiple times into PQueue wont come out more than once
    while(!pqueue.empty()){
      if(prioritymap.find(first.sa) != prioritymap.end()){

          // std::cout << first.sa << " " << first.priority << std::endl;

          if(prioritymap.find(first.sa)->second == first.priority){
              // std::cout << first.sa << " " << first.priority << std::endl;
              prioritymap.erase(first.sa);
              pqueue.pop();
              break;
          }else{
            pqueue.pop();
            first = pqueue.top();
          }

      }else{
          pqueue.pop();
          first = pqueue.top();
      }
    }

    // extraction S,A which has highest priority from PQueue
    int sampleState = atoi(first.sa.substr(1).c_str());
    int sampleAction = atoi(first.sa.substr(0,1).c_str());

    // std::cout << sampleState << std::endl;
    // std::cout << sampleAction << std::endl;

    // Get next state and reward from model map
    int sampleNextState = model[first.sa][0];
    int sampleReward = model[first.sa][1];

    // std::cout << sampleNextState << std::endl;
    // std::cout << sampleReward << std::endl;

    // update q_sa for the sample state
    int maxA = returnGreedyAction(q_sa, sampleNextState);
    double maxQ_SA = q_sa[stp1][maxA];
    
    q_sa[sampleState][sampleAction] += alpha*(reward + discount*maxQ_SA - q_sa[sampleState][sampleAction]);

    // extract all S,A that lead to sampleState
    std::set<std::string> set_SA = previous_sa[sampleState];

    for(std::set<std::string>::iterator itr = set_SA.begin(); itr != set_SA.end(); ++itr){
      std::string prev_sa = *itr;
      int prevState = atoi(prev_sa.substr(1).c_str());
      int prevAction = atoi(prev_sa.substr(0,1).c_str());

      // reward for prevState, prevAction, sampleState
      int prevReward = model[prev_sa][1];

      // q_sa for greedy action in sampleState
      maxA = returnGreedyAction(q_sa, sampleState);
      maxQ_SA = q_sa[sampleState][maxA];

      P = fabs(prevReward + discount*maxQ_SA - q_sa[prevState][prevAction]);

      if(P > theta){
        // insert into PQueue
        pqueue.push(saPair(prev_sa, P));

        // insert into prioritymap
        prioritymapInsert(prioritymap, prev_sa, P);
      }
    }

  }


  
  
  // printf(" %d-%d-%d-%d ", stp0, atp0, stp1, atp1);

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

  // get previous state and action that agent took to get to terminal state
  int stp0 = (int)gsl_vector_get(last_observation,0);
  int atp0 = (int)gsl_vector_get(last_action,0);

  int stp1 = 8;

  std::string sa0 = makeSAstring(stp0, atp0);

  if (model.find(sa0) == model.end()){

    // a vector containing next state and reward - this is value of model map
    std::vector<int> next_sr;
    next_sr.push_back(stp1);
    next_sr.push_back(reward);

    // insert S,A into model map
    model[sa0] = next_sr;
  }

  // q_sa[stp0][atp0] += alpha*(reward + discount*0 - q_sa[stp0][atp0]);

  double P = fabs(reward + discount*0 - q_sa[stp0][atp0]);

  // std::cout << P << std::endl;

  // int dummy;
  // std::cin >> dummy;

  if(P > theta){
    // insert into PQueue
    pqueue.push(saPair(sa0, P));

    // insert into prioritymap
    prioritymapInsert(prioritymap, sa0, P);
  }

  int planningStep = 0;

  while(planningStep < psteps && !pqueue.empty()){

    planningStep++;

    // take out the first element of PQueue
    saPair first = pqueue.top();

    // this loop ensures that the S,A pairs that are inserted multiple times into PQueue wont come out more than once
    while(!pqueue.empty()){
      if(prioritymap.find(first.sa) != prioritymap.end()){

          // std::cout << first.sa << " " << first.priority << std::endl;

          if(prioritymap.find(first.sa)->second == first.priority){
              prioritymap.erase(first.sa);
              pqueue.pop();
              break;
          }else{
            pqueue.pop();
            first = pqueue.top();
          }

      }else{
          pqueue.pop();
          first = pqueue.top();
      }
    }

    // extraction S,A which has highest priority from PQueue
    int sampleState = atoi(first.sa.substr(1).c_str());
    int sampleAction = atoi(first.sa.substr(0,1).c_str());

    // std::cout << sampleState << std::endl;
    // std::cout << sampleAction << std::endl;

    // Get next state and reward from model map
    int sampleNextState = model[first.sa][0];
    int sampleReward = model[first.sa][1];

    // std::cout << sampleNextState << std::endl;
    // std::cout << sampleReward << std::endl;

    // update q_sa for the sample state
    int maxA = returnGreedyAction(q_sa, sampleNextState);
    double maxQ_SA = q_sa[stp1][maxA];
    
    q_sa[sampleState][sampleAction] += alpha*(reward + discount*maxQ_SA - q_sa[sampleState][sampleAction]);

    // extract all S,A that lead to sampleState
    std::set<std::string> set_SA = previous_sa[sampleState];

    for(std::set<std::string>::iterator itr = set_SA.begin(); itr != set_SA.end(); ++itr){
      std::string prev_sa = *itr;
      int prevState = atoi(prev_sa.substr(1).c_str());
      int prevAction = atoi(prev_sa.substr(0,1).c_str());

      // reward for prevState, prevAction, sampleState
      int prevReward = model[prev_sa][1];

      // q_sa for greedy action in sampleState
      maxA = returnGreedyAction(q_sa, sampleState);
      maxQ_SA = q_sa[sampleState][maxA];

      P = fabs(reward + discount*maxQ_SA - q_sa[prevState][prevAction]);

      if(P > theta){
        // insert into PQueue
        pqueue.push(saPair(prev_sa, P));

        // insert into prioritymap
        prioritymapInsert(prioritymap, prev_sa, P);
      }
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
  //     printf("%d, ", total_result[i]/30);
  //   }
  // }

  // std::cout << std::endl;
  // printMap(model);

  model.clear();
  previous_sa.clear();
  prioritymap.clear();
  while(!pqueue.empty()){
    pqueue.pop();
  }

  // for(int i=0; i<60; i++){
  //   for(int j=0; j<4; j++){
  //     printf("%d - %f ", i, q_sa[i][j]);
  //   }
  //   printf("\n");
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
