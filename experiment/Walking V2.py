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
    game.set_doom_scenario_path("../scenarios/walking_linear_map.wad")
    game.set_doom_map("map01")

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
    game.set_episode_timeout(400)

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

# Select Region of Interest
def get_roi(img, verticies, value):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, verticies, 255)
    masked = cv.bitwise_and(img,mask)
    masked[masked == 0] = value
    return masked

def process_img(img):

    # Taking only the bottom half of the screen
    imgcopy = img[240:480, 0:640]

    left_blocks = []
    right_blocks = []
    centre_blocks = []
    inputs = []

    # Getting the region of interest
    verticies = np.array([[0, 240], [320, 0], [640, 240]])
    centre = get_roi(imgcopy, [verticies], 255)

    #Set up left side of screen
    left = imgcopy[0:240, 0:320]
    verticies = np.array([[0,0], [0,240], [320, 0]])
    left = get_roi(left, [verticies], 0)

    #Set up right side of screen
    right = imgcopy[0:240, 320:640]
    verticies = np.array([[0,0], [320,240], [320, 0]])
    right = get_roi(right, [verticies], 0)

    #Get left blocks
    left_blocks.append(left[0:240, 0:80])
    left_blocks.append(left[0:240, 80:160])
    left_blocks.append(left[0:240, 160:240])
    left_blocks.append(left[0:240, 240:320])

    #Get centre cone images
    centre_blocks.append(centre[200:240, 0:640])
    centre_blocks.append(centre[160:200, 0:640])
    centre_blocks.append(centre[120:160, 0:640])
    centre_blocks.append(centre[80:120, 0:640])
    centre_blocks.append(centre[40:80, 0:640])
    centre_blocks.append(centre[0:40, 0:640])

    # Get right blocks
    right_blocks.append(right[0:240, 240:320])
    right_blocks.append(right[0:240, 160:240])
    right_blocks.append(right[0:240, 80:160])
    right_blocks.append(right[0:240, 0:80])

    #Check each left area for a value less than 0.4
    for left_block in left_blocks:
        if np.amax(left_block, None, None, np._NoValue) < 14:
            inputs.append(1)
        else:
            inputs.append(0)

    for centre_block in centre_blocks:
        if 14 < np.amin(centre_block, None, None, np._NoValue) < 255:
            inputs.append(1)
        else:
            inputs.append(0)

    # Check each right area for a value less than 0.4
    for right_block in right_blocks:
        if np.amax(right_block, None, None, np._NoValue) < 14:
            inputs.append(1)
        else:
            inputs.append(0)

    # Return what will be input into the network
    return inputs

def fitness_function(genomes, config):
    global angle
    for genome_id, genome in genomes:
        genome.fitness = 0

        # Create the neural network as defined by the genes
        net = nn.FeedForwardNetwork.create(genome, config)

        # Start new episode
        game.new_episode()

        # Resetting the state of the agent
        is_still = False
        is_still_tic = 0

        # Setting the target position and setting the closest distance to some large value
        target_x = 1568
        closest_distance = 9999;

        # Run episode until timeout or the agent makes no action
        while xor(not game.is_episode_finished(), is_still):
            # Get the state variables
            state = game.get_state()
            depth_buffer = state.depth_buffer
            angle = state.game_variables[2]

            # Get the inputs for the network
            inputs = process_img(depth_buffer)

            # Get the output from the network
            outputs =  net.activate(inputs)

            # Make an action based of the outputs and get the reward if any for that action
            reward = game.make_action(outputs)

            # Check whether the agent is making an action
            if xor(outputs[0] != 0, outputs[1] != 0) or outputs[2] != 0:
                is_still_tic = 0
            else:
                is_still_tic += 1

            # Checking whether the agent has stopped
            if is_still_tic >= 10:
                is_still = True

            # Getting the position of the agent after its action has been made
            character_position_x = state.game_variables[0]

            # Add the reward from the action to the genomes fitness
            genome.fitness += reward

            # Calculating the closest the agent has been to the target location
            distance = math.sqrt(((math.fabs(target_x - character_position_x)) ** 2))
            if distance < closest_distance:
                closest_distance = distance

        # Calculating the fitness of the genome including the distance and any rewards for actions
        genome.fitness = genome.fitness +  (1888 - closest_distance)

        # If the agent makes no movement, calculate how much the agent has turned as it fitness
        if genome.fitness == 0:
            genome.fitness = abs(np.round(180-angle))

        # Displaying the fitness of the current genome
        print("Genome: {!r} fitness: {!r}".format(genome_id, genome.fitness))

# Save a genome
def save_genome(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def run_scenario():
    # Create configuration object
    config = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet, stagnation.DefaultStagnation, "../config files/config_walking_V2")

    # Set maximum number of episodes
    episodes = 50

    # Create a population of genomes from the set configuration
    pop = Population(config)

    # Add reporters to keep track of performance of the genomes
    pop.add_reporter(StdOutReporter(True))
    stats = StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(Checkpointer(5, 5, "../checkpoints/walking2/walking-V2.2-checkpoint-"))

    # Run NEAT and retrieve the highest performing genome
    winner = pop.run(fitness_function, episodes)

    # Save the highest performing genome
    save_genome(winner, "moving-genome-PT2.2")

    # Display the fitness of the highest performing genome
    print("highest fitness value: {!r}".format(winner.fitness))

    # Display the genome's network structure and fitness and species graphs
    node_names = {-9: 'Right 1', -1: 'Left 1', -8: 'Right 2', -2: 'Left 2', -7: 'Right 3', -3: 'Left 3', -6: 'Right 4',
                  -4: 'Left 4', -5: 'Centre', 0: 'Turn right', 1: 'Turn left', 2:'Move Forward'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # Close game once complete
    game.close()

run_scenario()