from __future__ import print_function
from neat import *
from vizdoom import *
import skimage.color, skimage.transform
import numpy as np
from operator import xor
import visualize
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import cv2 as cv
import pickle

def init_game():
    game = DoomGame()

    # Set scenario and map
    game.set_doom_scenario_path("../scenarios/twoEnemies.wad")
    game.set_doom_map("map01")

    # Set screen format
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.GRAY8)

    # Enabling buffers
    game.set_automap_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    # Setting available buttons
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)

    # Setting game variables
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.ANGLE)

    # Setting timeout in ticks (actions)
    game.set_episode_timeout(200)

    # Setting episode to start after 10 ticks
    game.set_episode_start_time(10)

    game.set_render_weapon(False)

    # Window visability
    game.set_window_visible(True)

    # Living reward
    game.set_living_reward(0)

    # Setting gamemode
    game.set_mode(Mode.PLAYER)

    game.init()

    return game

game = init_game()
resolution = (3,32)

def process_img(img):
    # Resize image
    img = skimage.transform.resize(img, resolution, mode='constant')

    left_blocks = []
    right_blocks = []

    # Split the screen into 9 regions
    left_blocks.append(img[0:3, 0:8])
    left_blocks.append(img[0:3, 8:12])
    left_blocks.append(img[0:3, 12:14])
    left_blocks.append(img[0:3, 14:15])
    centre_block = img[0:3, 15:17]
    right_blocks.append(img[0:3, 17:18])
    right_blocks.append(img[0:3, 18:20])
    right_blocks.append(img[0:3, 20:24])
    right_blocks.append(img[0:3, 24:32])

    inputs = []

    # Check whether there is a target to move towards in all left regions of the screen
    for left_block in left_blocks:
        if np.amax(left_block, None, None, np._NoValue) > 0:
            inputs.append(1)
        else:
            inputs.append(0)

    # Check whether there is a target to move towards in the centre of the screen.
    if np.amax(centre_block, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    # Check whether there is a target to move towards in all left regions of the screen
    for right_block in right_blocks:
        if np.amax(right_block, None, None, np._NoValue) > 0:
            inputs.append(1)
        else:
            inputs.append(0)

    # Return what will be input into the network
    return inputs

def fitness_function(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0

        # Create the neural network as described by the genome
        net = nn.FeedForwardNetwork.create(genome, config)

        # Resetting the state for the agent
        left_complete = False
        right_complete = False

        # attempt until episode is complete with the enemy spawning either side of the agent
        while not right_complete or not left_complete:

            # Start new episode
            game.new_episode()

            # Get the state variables
            state = game.get_state()

            # Setting the episode to not run initially
            run_episode = False

            # Get the position of the enemy
            if len(state.labels) != 0:
                enemy_position_x = state.labels[0].object_position_x
                enemy_position_y = state.labels[0].object_position_y

            # Check the position of the enemy, only running if corresponding episode has not already run
            if (enemy_position_x == -128 and enemy_position_y == 288) and not left_complete:
                run_episode = True
                left_complete = True
            elif (enemy_position_x == -128 and enemy_position_y >= -224) and not right_complete:
                run_episode = True
                right_complete = True

            # Only run the episode if needed
            if run_episode:

                # Initialising still counter
                is_still = False
                is_still_counter = 0

                # Run episode until timeout or the agent doesn't perform an action
                while xor(not game.is_episode_finished(), is_still):
                    # Get the state variables
                    state = game.get_state()
                    label_buff = state.labels_buffer
                    angle = state.game_variables[2]

                    # Get inputs for the neural network
                    inputs = process_img(label_buff)

                    # Get outputs from the neural network
                    outputs = net.activate(inputs)

                    # Checking the agent is making any action
                    if xor(outputs[0] != 0, outputs[1] != 0) or outputs[2] != 0:
                        is_still_counter = 0
                    else:
                        is_still_counter += 1

                    # Checking whether the agent has stopped
                    if is_still_counter >= 10:
                        is_still = True

                    # Make action and get any reward for the action performed
                    genome.fitness += game.make_action(outputs)

                # Calculate the angle in relation to the position of the enemy
                if enemy_position_x == -128 and enemy_position_y == 288:
                    if angle <= 45:
                        genome.fitness += angle
                    elif angle > 45 and angle <= 90:
                        genome.fitness += (90 - angle)
                    elif angle > 90:
                        genome.fitness += 0
                elif enemy_position_x == -128 and enemy_position_y >= -224:
                    if angle <= 315 and angle > 270:
                        genome.fitness += angle - 270
                    elif angle > 315 and angle <= 360:
                        genome.fitness += (360 - angle)
                    elif angle < 315:
                        genome.fitness += 0

        # Take an average of the genomes fitness
        genome.fitness = genome.fitness / 2

        # Print the genome's fitness score
        print("Genome: {!r} Fitness: {!r}".format(genome_id, genome.fitness))

# Save the desired genome
def save_genome(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def run_scenario():
    # Create configuration object
    config = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet, stagnation.DefaultStagnation, "../config files/config_shoot_enemy")

    # Create a population of genomes from the set configuration
    pop = population.Population(config)
    
    # Set maximum number of episodes
    episodes = 50

    # Add reporters to keep track of performance of the genomes
    pop.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(Checkpointer(5, 5, "shoot-enemy-checkpoint-"))

    # Run NEAT and retrieve the highest performing genome
    winner = pop.run(fitness_function, episodes)

    # Display the fitness of the highest performing genome
    print("highest fitness value: {!r}".format(winner.fitness))

    # Display the genome's network structure and fitness and species graphs
    node_names = {-9:'Right 1', -1:'Left 1', -8:'Right 2', -2:'Left 2', -7:'Right 3', -3:'Left 3', -6:'Right 4', -4:'Left 4', -5:'Centre', 0:'Turn right', 1:'Turn left', 2:'Shoot'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Save best performing genome
    save_genome(winner, "shooting-genome")

    # Close game once complete
    game.close()

run_scenario()