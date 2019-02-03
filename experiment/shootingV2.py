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

    #Set scenario and map
    game.set_doom_scenario_path("../scenarios/twoEnemies.wad")
    game.set_doom_map("map01")

    #Set screen format
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.GRAY8)

    #Enabling buffers
    game.set_automap_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    #Setting available buttons
    #game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.ATTACK)

    #Setting game variables
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.ANGLE)

    #Setting timeout in ticks (actions)
    game.set_episode_timeout(200)

    #Setting episode to start after 10 ticks
    game.set_episode_start_time(10)

    game.set_render_weapon(False)

    #Window visability
    game.set_window_visible(True)

    #Living reward
    game.set_living_reward(0)

    #Setting gamemode
    game.set_mode(Mode.PLAYER)

    game.init()

    return game

game = init_game()
resolution = (5,64)

def process_img(img):

    img = skimage.transform.resize(img, resolution, mode="constant")

    # Split the image into left, right and centre regions
    left = img[0:5, 0:30]
    centre = img[0:5, 30:34]
    right = img[0:5, 34:64]

    inputs = []

    # Check whether there is a target in the left region of the screen
    if np.amax(left, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    # Check whether there is a target in the centre region of the screen
    if np.amax(centre, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    # Check whether there is a target in the right region of the screen
    if np.amax(right, None, None, np._NoValue) > 0:
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

        # Resetting the state of the agent
        left_complete = False
        right_complete = False

        # Attempt until episode is complete with an enemy spawning on either side of the agent
        while not right_complete or not left_complete:

            # Start net episode
            game.new_episode()

            # Get the game state variables
            state = game.get_state()

            # Setting the episode to not run initially
            run_episode = False

            # Getting the position of the agent
            if len(state.labels) != 0:
                enemy_position_x = state.labels[0].object_position_x
                enemy_position_y = state.labels[0].object_position_y

            # Checking the position of the enemy, only running if corresponding episode has not already run
            if (enemy_position_x == -128 and enemy_position_y == 288) and not left_complete:
                run_episode = True
                left_complete = True
            elif (enemy_position_x == -128 and enemy_position_y >= -224) and not right_complete:
                run_episode = True
                right_complete = True

            # Only run the episode if needed
            if run_episode:

                # Initialising the state of the agent
                is_still = False
                is_still_counter = 0

                # Run episode until timeout or the agent doesn't perform an action
                while xor(not game.is_episode_finished(), is_still):
                    # Get the state variables
                    state = game.get_state()
                    label_buff = state.labels_buffer
                    angle = state.game_variables[2]

                    # Get the inputs for the network
                    inputs = process_img(label_buff)

                    # Get outputs from the network
                    outputs = net.activate(inputs)

                    # Check whether the agent has made an action
                    if xor(outputs[0] != 0, outputs[1] != 0) or outputs[2] != 0:
                        is_still_counter = 0
                    else:
                        is_still_counter += 1

                    # Check whether the agent has stopped
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

        # Take average of the genomes fitness
        genome.fitness = genome.fitness / 2

        # Print the genome's fitness
        print("Genome: {!r} Fitness: {!r}".format(genome_id, genome.fitness))

# Save a genome
def save_genome(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def run_scenario():
    # Create configuration object
    config = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet, stagnation.DefaultStagnation, "../config files/config_shooting_pt2")

    # Create a population of genomes from the set configuration
    pop = population.Population(config)

    # Set maximum number of episodes
    episodes = 50

    # Add reporters tp keep track of performance of the genomes
    pop.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(Checkpointer(5, 5, "../checkpoints/shootingV2/shoot-enemy-checkpoint-"))

    # Run NEAT and retrieve the highest performing genome
    winner = pop.run(fitness_function, episodes)

    # Print the fitness of the highest performing genome
    print("highest fitness value: {!r}".format(winner.fitness))

    # Display the genome's network structure and fitness and species graphs
    node_names = {-1:'Left ', -2:'Centre 2',  -3:'Right 3', 0:'Turn right', 1:'Turn left', 2:'Shoot'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Save highest performing genome
    save_genome(winner, "shooting-genome")

    # Close game once complete
    game.close()

run_scenario()