from firasim_client.env import MarkovDecisionProcess

if __name__ == "__main__":
    mdp = MarkovDecisionProcess()
    state = mdp.reset_random_init_pos()
    while True:
        print(state)
        packet = mdp.vision.read()
        if packet is not None:
            print(packet)
    # mdp.step([[0.5, 0.5]])