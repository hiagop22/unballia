from firasim_client.env import MarkovDecisionProcess

if __name__ == "__main__":
    mdp = MarkovDecisionProcess()
    state = mdp.reset_random_init_pos()
    print(state)