from __future__ import print_function
from neat import *
from vizdoom import *
import skimage.color, skimage.transform
import numpy as np
from operator import xor
import visualize
import math
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2 as cv

def init_game():
    game = DoomGame()

    #Set scenario and map
    game.set_doom_scenario_path("../scenarios/walkingPT2.1.wad")
    game.set_doom_map("map01")

    #game.set_doom_scenario_path("../scenarios/redemption.wad")
    #game.set_doom_map("E1M1")

    #Set screen format
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.GRAY8)

    #Enabling buffers
    game.set_automap_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    #Setting available buttons
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.MOVE_FORWARD)

    #Setting game variables
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.ANGLE)

    #Setting timeout in ticks (actions)
    game.set_episode_timeout(600)

    #Setting episode to start after 10 ticks
    game.set_episode_start_time(10)

    #Window visability
    game.set_window_visible(True)

    #Living reward
    game.set_living_reward(0)

    #Setting gamemode
    game.set_mode(Mode.PLAYER)

    game.init()

    return game

game = init_game()
resolution = (3,64)

def process_img(img):
    # Resize image to defined resolution
    img = skimage.transform.resize(img, resolution, mode='constant')

    left_blocks = []
    right_blocks = []
    
    # Split the screen into 9 regions
    left_blocks.append(img[0:3, 0:16])
    left_blocks.append(img[0:3, 16:24])
    left_blocks.append(img[0:3, 24:28])
    left_blocks.append(img[0:3, 28:30])
    centre_block = img[0:3, 30:34]
    right_blocks.append(img[0:3, 34:36])
    right_blocks.append(img[0:3, 36:40])
    right_blocks.append(img[0:3, 40:48])
    right_blocks.append(img[0:3, 48:64])

    inputs = []

    distance_threshold = 0.4

    # Check all left regions whether the agent is close enough to a wall/object
    for left_block in left_blocks:
        if np.amin(left_block, None, None, np._NoValue) < distance_threshold:
            inputs.append(1)
        else:
            inputs.append(0)

    # Check whether the agent is close enough to a wall/object
    if np.amin(centre_block, None, None, np._NoValue) > distance_threshold:
        inputs.append(1)
    else:
        inputs.append(0)

    # Check all right regions whether the agent is close enough to a wall/object
    for right_block in right_blocks:
        if np.amin(right_block, None, None, np._NoValue) < distance_threshold:
            inputs.append(1)
        else:
            inputs.append(0)

    # Return what will be input into the network
    return inputs

def fitness_function(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0

        # Create the neural network as defined by the genes
        net = nn.FeedForwardNetwork.create(genome, config)

        # Initialising the state of the agent
        is_still = False
        is_still_tic = 0

        # Setting the target position and setting the closest distance to the distance between the agent and the target
        target_x = 1344
        target_y = 32
        closest_distance = 1728

        # Start new episode
        game.new_episode()

        # Run episode until timeout or the agent makes no action
        while xor(not game.is_episode_finished(), is_still):
            # Get the state variables
            state = game.get_state()
            depth_buffer = state.depth_buffer

            # Get the inputs for the network
            inputs = process_img(depth_buffer)

            # Get the output from the network
            outputs = net.activate(inputs)

            # Make an action based of outputs of the network
            game.make_action(outputs)

            # Check whether the agent is making an action
            if (xor(outputs[0] != 0, outputs[1] != 0)) or outputs[2] != 0:
                is_still_tic = 0
            else:
                is_still_tic += 1

            # Checking whether the agent has stopped
            if is_still_tic >= 10:
                is_still = True

            # Getting position of the agent after it has made an action
            character_position_x = state.game_variables[0]
            character_position_y = state.game_variables[1]

            # Calculating the closest the agent has come to the target position
            distance = math.sqrt(((math.fabs(target_x -character_position_x)) ** 2) + ((math.fabs(target_y - character_position_y)) **2))
            if distance < closest_distance:
                closest_distance = distance

        # Calculating the genomes fitness
        genome.fitness = 1728 - closest_distance

        # Setting the fitness of the genome to be 1 if the agent walks into the wall at the end of the corridor
        if genome.fitness == 559.9755859375:
            genome.fitness = 1

        # Display the fitness of the current genome
        print("Genome: {!r} fitness: {!r}".format(genome_id, genome.fitness))

# Save a genome
def save_genome(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def run_scenario():
    # Create configuration object
    config = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet, stagnation.DefaultStagnation, "../config files/config_traverse")

    # Create a population of genomes using the configuration
    pop = Population(config)

    # Set maximum number of episodes
    episodes = 50

    # Add reporters to keep track of the performance of the genomes
    pop.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(Checkpointer(5, 5, "../checkpoints/walking2/walking-checkpoint-"))

    # Run NEAT and retrieve the highest performing genomes
    winner = pop.run(fitness_function, episodes)

    # Save the highest performing genome
    save_genome(winner, "moving-genome")

    # Display the fitness of the highest performing genome
    print("highest fitness value: {!r}".format(winner.fitness))

    # Display the genome's network structure and fitness and species graphs
    node_names = {-9: 'Right 1', -1: 'Left 1', -8: 'Right 2', -2: 'Left 2', -7: 'Right 3', -3: 'Left 3', -6: 'Right 4',
                  -4: 'Left 4', -5: 'Centre', 0: 'Turn right', 1: 'Turn left', 2:'Move Forward'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Close the game once complete
    game.close()

run_scenario()