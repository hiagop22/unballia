import time
from firasim_client.env import MarkovDecisionProcess

if __name__ == "__main__":
    loop_time = 0.02
    mdp = MarkovDecisionProcess()
    state = mdp.reset_random_init_pos()
    print(state)
    
    last = time.time()
    last_packet = None

    while True:
        if (time.time() - last) < loop_time:
            packet = mdp.vision.read()
            if packet is not None:
                last_packet = packet
        else:
            break

    # print(last_packet)
    next_state, reward, done = mdp.step([[0.2, 0.2]])
    print(next_state)
    print(reward)
    print(done)