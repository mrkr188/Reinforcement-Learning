#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// printing vector
void printVector(vector<double>& v){
    for (vector<double>::const_iterator i = v.begin(); i != v.end(); ++i)
        cout << *i << ' ';
    cout<< endl;
}

void printVector(vector<int>& v){
    for (vector<int>::const_iterator i = v.begin(); i != v.end(); ++i)
        cout << *i << ' ';
    cout<< endl;
}


int main()
{
    
    // final goal 
	int goal = 100;
    
    // a vector storing state values
    // a vector storing states. states 0 to 100. 
	// 1-99 normal states.terminal states 0 is lose & 100 is win.
    vector<double> state_values(goal+1, 0);
    state_values[100] = 1;
    
    // a vector storing optimal policy
    vector<int> policy(goal+1, 0);
    
    // probability of head
    double p_head = 0.4;
	
	
	// delta and epislon used to terminate loop
	double delta;
	double epislon = 0.00000000000000000000000000000000001;
	
	// count number of iterations
	int iter_count = 0;
	
	// value iteration
	while(1){
	    
	    iter_count++;
	    cout<< "iteration " << iter_count << endl;
	    delta = 0.0;
	    
	    // iterate each state to find value of state
	    // iterate from state 1-99. 0 and 99 are terminal states
	    for(int state=1; state<100; ++state){
	        
	        // number of actions possible for current state. actions 1 to min(state, goal-state)
	        int num_actions = min(state, goal - state);
	        
	        // a vector to store action values for each actions
	        // in action_values vector, i take indexes 1 - num_actions. 0 index is left alone
	        vector<double> action_values(num_actions + 1);
	        
	        // loop all actions to find action values
	        for(int action=1; action<=num_actions; ++action){
	            action_values[action] = p_head * state_values[state + action] + (1 - p_head) * state_values[state - action];
	        }
	        
	        // new state value is max of action values
	        // here i iterate from index 1 to num_actions
	        double new_state_value = *max_element(begin(action_values) + 1, end(action_values));
	        
	        // update delta
	        delta = max(0.0, abs(new_state_value - state_values[state]));
	        
	        // update state value
	        state_values[state] = new_state_value;
	        
	    }
	    
	    cout << delta << endl;
	    
	    if(delta < epislon){
	    	break;
	    }
	    
	}
	
	cout << "\n\nState Values" << endl;
	printVector(state_values);
	
	// optimal policy
	for(int state=1; state<100; ++state){
		
			// number of actions possible for current state
	        int num_actions = min(state, goal - state);
	        
	        // a vector to store action values for each actions
	        vector<double> action_values(num_actions + 1);
	        
	        // loop all actions to find action values
	        for(int action=1; action<=num_actions; ++action){
	            action_values[action] = p_head * state_values[state + action] + (1 - p_head) * state_values[state - action];
	        }
	        
	        // assign greedy policy
	        policy[state] = max_element(begin(action_values)+1, end(action_values)) - begin(action_values);
	        
	}
	
	
	cout << "\n\nPolicies" << endl;
	printVector(policy);
	
	
}
