import time

from verify_deep_q_learning_with_replay_with_reward_shaping import \
    verify_deep_q_learning_with_replay_with_reward_shaping
from Frozen_Lake_Environment import Frozen_Lake_Environment
from verify_double_deep_v1 import verify_deep_q_learning_with_target_network
from verify_double_deep_v2 import verify_double_deep_2015


runs = ['verify_deep_q_learning_with_target_network']


if __name__ == '__main__':




    if 'verify_double_deep_2015' in runs:
        print("+------------------------------------------+")
        print("|                                          |")
        print("|       STARTING DOUBLE DEEP Q-ALGO        |")
        print("|            Replay buffer: Yes            |")
        print("|          Convolution layer: No           |")
        print("|                                          |")
        print("+------------------------------------------+")
        print()

        # Length\Width of square-shaped board
        size = 3
        env2 = Frozen_Lake_Environment(size)
        env = Frozen_Lake_Environment(size)
        env.print_environment_parameters()
        env2.print_environment_parameters()
        env.add_layout_3_1()
        env2.add_layout_3_1()
        env.return_all_down_critical_states()
        env.print_environment_parameters()
        env.print_on_board_current_state()
        agent = verify_double_deep_2015(env, size, env2)
        apply_verification_fix = True
        start_algo = time.time()
        agent.train(1000, apply_verification_fix)
        start = []
        for i in range(size * size - 1):
            if i not in env.hole_index_list:
                start.append(i)
        print(f'Running entire algorithm took: {time.time() - start_algo} seconds.')
        print()
        print("Testing:")
        total = 0
        total_suc = 0
        for i in start:
            print("------------------------")
            total = total + 1
            if agent.test_model(agent.q_network_init, i):
                total_suc = total_suc + 1
        print("Suc rate: ", total_suc * 100 / total)
        agent.plot_learning_curve(apply_verification_fix)

        apply_verification_fix = False
        start_algo = time.time()
        agent.train(200000, apply_verification_fix)
        start = []
        for i in range(size * size - 1):
            if i not in env.hole_index_list:
                start.append(i)
        print(f'Running entire algorithm took: {time.time() - start_algo} seconds.')
        print()
        print("Testing:")
        total = 0
        total_suc = 0
        for i in start:
            print("------------------------")
            total = total + 1
            if agent.test_model(agent.q_network_init, i):
                total_suc = total_suc + 1
        print("Suc rate: ", total_suc * 100 / total)
        agent.plot_learning_curve(apply_verification_fix)
    if 'verify_deep_q_learning_with_target_network' in runs:
        print("+------------------------------------------+")
        print("|                                          |")
        print("|       STARTING DOUBLE DEEP Q-ALGO        |")
        print("|            Replay buffer: Yes            |")
        print("|          Convolution layer: No           |")
        print("|                                          |")
        print("+------------------------------------------+")
        print()

        # Length\Width of square-shaped board
        size = 3
        env2 = Frozen_Lake_Environment(size)
        env = Frozen_Lake_Environment(size)
        env.print_environment_parameters()
        env2.print_environment_parameters()
        env.add_layout_3_2()
        env2.add_layout_3_2()
        env.return_all_down_critical_states()
        env.print_environment_parameters()
        env.print_on_board_current_state()
        agent = verify_deep_q_learning_with_target_network(env, size, env2)
        apply_verification_fix = True
        start_algo = time.time()
        agent.train(1000, apply_verification_fix)
        start = []
        for i in range(size * size - 1):
            if i not in env.hole_index_list:
                start.append(i)
        print(f'Running entire algorithm took: {time.time() - start_algo} seconds.')
        print()
        print("Testing:")
        total = 0
        total_suc = 0
        for i in start:
            print("------------------------")
            total = total + 1
            if agent.test_model(agent.q_network_init, i):
                total_suc = total_suc + 1
        print("Suc rate: ", total_suc * 100 / total)
        agent.plot_learning_curve(apply_verification_fix)

        apply_verification_fix = False
        start_algo = time.time()
        agent.train(200000, apply_verification_fix)
        start = []
        for i in range(size * size - 1):
            if i not in env.hole_index_list:
                start.append(i)
        print(f'Running entire algorithm took: {time.time() - start_algo} seconds.')
        print()
        print("Testing:")
        total = 0
        total_suc = 0
        for i in start:
            print("------------------------")
            total = total + 1
            if agent.test_model(agent.q_network_init, i):
                total_suc = total_suc + 1
        print("Suc rate: ", total_suc * 100 / total)
        agent.plot_learning_curve(apply_verification_fix)



    if 'verify_deep_q_learning_with_replay_with_reward_shaping' in runs:
        print("+------------------------------------------+")
        print("|                                          |")
        print("|       STARTING SIMPLE DEEP Q-ALGO        |")
        print("|            Replay buffer: Yes            |")
        print("|          Convolution layer: No           |")
        print("|                                          |")
        print("+------------------------------------------+")
        print()



        # Length\Width of square-shaped board
        size = 3
        env2 = Frozen_Lake_Environment(size)
        env = Frozen_Lake_Environment(size)
        env.print_environment_parameters()
        env2.print_environment_parameters()
        env.add_layout_3_1()
        env2.add_layout_3_1()
        env.return_all_down_critical_states()
        env.print_environment_parameters()
        env.print_on_board_current_state()
        agent = verify_deep_q_learning_with_replay_with_reward_shaping(env, size,env2)
        apply_verification_fix = True
        start_algo = time.time()
        agent.train(1500, apply_verification_fix)
        start = []
        for i in range(size * size - 1):
            if i not in env.hole_index_list:
                start.append(i)
        #print(f'Running entire algorithm took: {time.time()-start_algo} seconds.')
        #print()
        print("Testing with verification:")
        total = 0
        total_suc = 0
        for i in start:
            print("------------------------")
            total = total + 1
            if agent.test_model(agent.q_network_init, i):
                total_suc = total_suc + 1
        print("Suc rate: ", total_suc * 100 / total)
        agent.plot_learning_curve(apply_verification_fix)


        apply_verification_fix = False
        start_algo = time.time()
        agent.train(10000000, apply_verification_fix)
        start = []
        for i in range(size * size - 1):
            if i not in env.hole_index_list:
                start.append(i)
        #print(f'Running entire algorithm took: {time.time() - start_algo} seconds.')
        #print()
        print("Testing without verification:")
        total = 0
        total_suc = 0
        for i in start:
            print("------------------------")
            total = total + 1
            if agent.test_model(agent.q_network_init, i):
                total_suc = total_suc + 1
        print("Suc rate: ", total_suc * 100 / total)
        agent.plot_learning_curve(apply_verification_fix)

