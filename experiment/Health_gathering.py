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
    game.set_doom_scenario_path("../scenarios/health_two_pack.wad")
    game.set_doom_map("map01")

    # Set screen format
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_screen_format(ScreenFormat.GRAY8)

    # Enabling buffers
    game.set_automap_buffer_enabled(True)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    # Setting available buttons
    # game.add_available_button(Button.TURN_LEFT_RIGHT_DELTA)
    game.add_available_button(Button.TURN_RIGHT)
    game.add_available_button(Button.TURN_LEFT)
    game.add_available_button(Button.MOVE_FORWARD)

    # Setting game variables
    game.add_available_game_variable(GameVariable.POSITION_X)
    game.add_available_game_variable(GameVariable.POSITION_Y)
    game.add_available_game_variable(GameVariable.HEALTH)
    game.add_available_game_variable(GameVariable.ANGLE)

    # Setting timeout in ticks (actions)
    game.set_episode_timeout(150)

    # Setting episode to start after 10 ticks
    game.set_episode_start_time(10)

    # Window visabilityx
    game.set_window_visible(True)

    # Living reward
    game.set_living_reward(1)

    # Setting gamemode
    game.set_mode(Mode.PLAYER)

    game.init()

    return game

game = init_game()
resolution = (6,64)

def process_img(img):
    # Change resolution of the image to more approriate one
    img = skimage.transform.resize(img, resolution, mode='constant')

    left_blocks = []
    right_blocks = []

    # Split the screen into 9 regions
    left_blocks.append(img[0:6, 0:16])
    left_blocks.append(img[0:6, 16:24])
    left_blocks.append(img[0:6, 24:28])
    left_blocks.append(img[0:6, 28:30])
    centre_block = img[0:6, 30:34]
    right_blocks.append(img[0:6, 34:36])
    right_blocks.append(img[0:6, 36:40])
    right_blocks.append(img[0:6, 40:48])
    right_blocks.append(img[0:6, 48:64])

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

        # Create the neural network as defined by the genes.
        net = nn.FeedForwardNetwork.create(genome, config)

        # Setting the number of attempts to be made by the genome
        attempts = 4

        # Running the scenario many times for each genome
        for i in range(0, attempts, 1):

            # Start new episode
            game.new_episode()
            is_still = False
            is_still_counter = 0

            # initalising state variables
            health_position_x = 0
            health_position_y = 0
            character_position_x = 0
            character_position_y = 0
            angle = 0

            # Run episode until timeout or until the agent makes no movement
            while xor(not game.is_episode_finished(), is_still):
                state = game.get_state()
                label_buff = state.labels_buffer

                # Generate inputs to the network
                inputs = process_img(label_buff)

                # Get the angle of the agent within the world
                angle = state.game_variables[3]

                # getting position once in sight.
                if len(state.labels) != 0:
                    health_position_x = state.labels[0].object_position_x
                    health_position_y = state.labels[0].object_position_y

                # Generate outputs from the inputs using the previously created network
                outputs = net.activate(inputs)

                # Check to see whether the agent is making any action
                if xor(outputs[0] != 0, outputs[1] != 0) or outputs[2] != 0:
                    is_still_counter = 0
                else:
                    is_still_counter += 1

                # End the episode if the agent is still
                if is_still_counter >= 40:
                    is_still = True

                # Get the position of the agent within the world
                character_position_x = state.game_variables[0]
                character_position_y = state.game_variables[1]

                # Make an action
                game.make_action(outputs)

            # calculate the angle in relation to the position of the health pack.
            if health_position_x == 864 and health_position_y == 864:
                if angle < 45:
                    genome.fitness += angle
                elif angle > 45 and angle < 90:
                    genome.fitness += (90 - angle)
                elif angle > 90:
                    genome.fitness += 0
            elif health_position_x == 864 and health_position_y == 352:
                if angle < 315 and angle > 270:
                    genome.fitness += angle - 270
                elif angle > 315 and angle < 360:
                   genome.fitness += (360 - angle)
                elif angle < 315:
                    genome.fitness += 0

            # Calculate the distance between the health and the agent
            distance = math.sqrt(((health_position_x - character_position_x) ** 2) + ((health_position_y - character_position_y) ** 2))
            genome.fitness += 362 - distance

        # Calculate an average fitness
        genome.fitness = genome.fitness / attempts

        # Print the fitness of the genome
        print("Genome: {!r} Fitness: {!r}".format(genome_id, genome.fitness))

# Save the best performing genome
def save_genome(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def run_scenario():
    # Create configuration object
    config = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet, stagnation.DefaultStagnation, "../config files/config_health_gathering")

    # Create a population using the configuation
    pop = population.Population(config)

    # Set maximum number of episodes
    episodes = 25

    # Add reporters to keep track of the performance of the genomes.
    pop.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(Checkpointer(5, 5, "multi-health-9-2-noBias-checkpoint-"))

    #  Run NEAT and retrieve the highest performing genome
    winner = pop.run(fitness_function, episodes)

    # Display the fitness of the highest performing network
    print("highest fitness value: {!r}".format(winner.fitness))

    # Display the genome's network structure and fitness and species graphs
    node_names = {-9:'Right 1', -1:'Left 1', -8:'Right 2', -2:'Left 2', -7:'Right 3', -3:'Left 3', -6:'Right 4', -4:'Left 4', -5:'Centre', 0:'Turn right', 1:'Turn left', 2:'Move Forward'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Save best performing genome
    save_genome(winner, "Health-gathering-genome")

    # Close game once complete
    game.close()

# Run the scenario using only the highest performing genome
def run_winner(winner, config):

    attempts = 20

    for attempt in range(0,attempts,1):
        net = nn.FeedForwardNetwork.create(winner, config)

        game.new_episode()
        is_still = False
        is_Still_counter = 0

        while xor(not game.is_episode_finished(), is_still):
            state = game.get_state()
            img = state.labels_buffer

            inputs = process_img(img)

            outputs = net.activate(inputs)

            game.make_action(outputs)

            if xor(outputs[0] != 0, outputs[1] != 0) or outputs[2] != 0:
                is_Still_counter = 0
            else:
                is_Still_counter += 1

            if outputs[2] != 0:
                is_Still_counter = 0
            else:
                is_Still_counter += 1

            if is_Still_counter >= 25:
                is_still = True

run_scenario()