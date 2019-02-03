#!/usr/bin/env python

from __future__ import print_function

from random import choice, random
from time import sleep, time
from vizdoom import *
import skimage.color, skimage.transform
import numpy as np
from operator import xor
from neat import *
import pickle
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

# For multiplayer game use process (ZDoom's multiplayer sync mechanism prevents threads to work as expected).
from multiprocessing import cpu_count, Process

# For singleplayer games threads can also be used.
# from threading import Thread

# Config
episodes = 1
timelimit = 2 # minutes
players = 6 # number of players

skip = 4
mode = Mode.PLAYER # or Mode.ASYNC_PLAYER
ticrate = 1 * DEFAULT_TICRATE # for Mode.ASYNC_PLAYER
random_sleep = True
const_sleep_time = 0.005
window = True
resolution = ScreenResolution.RES_256X144

args =""
console = False
config = "../Game Configs/cig.cfg"

# Setting the resolution for the different input representations
traverse_resolution = (1,64)
item_resolution = (6,64)
shooting_resolution = (5,64)

# Select Region of Interest
def get_roi(img, verticies, value):
    mask = np.zeros_like(img)
    cv.fillPoly(mask, verticies, 255)
    masked = cv.bitwise_and(img,mask)
    masked[masked == 0] = value
    return masked

# Process image for movement
def process_img_traverse(img):

    imgcopy = img[240:480, 0:640]

    left_blocks = []
    right_blocks = []
    centre_blocks = []
    inputs = []

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

    return inputs

# Process image for item gathering
def process_img_item_gather(img):
    img = skimage.transform.resize(img, (480, 640), mode='constant')

    img = skimage.transform.resize(img, item_resolution, mode='constant')

    left_blocks = []
    right_blocks = []

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

    for left_block in left_blocks:
        if np.amax(left_block, None, None, np._NoValue) > 0:
            inputs.append(1)
        else:
            inputs.append(0)

    if np.amax(centre_block, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    for right_block in right_blocks:
        if np.amax(right_block, None, None, np._NoValue) > 0:
            inputs.append(1)
        else:
            inputs.append(0)

    return inputs

# Process image for shooting
def process_img_shooting(img):
    imgcopy = skimage.transform.resize(img, shooting_resolution, mode="constant")

    left = imgcopy[0:5, 0:30]
    centre = imgcopy[0:5, 30:34]
    right = imgcopy[0:5, 34:64]

    inputs = []

    if np.amax(left, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    if np.amax(centre, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    if np.amax(right, None, None, np._NoValue) > 0:
        inputs.append(1)
    else:
        inputs.append(0)

    return inputs

def setup_player():
    game = DoomGame()

    game.load_config(config)
    game.set_mode(mode)
    game.add_game_args(args)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_console_enabled(console)
    game.set_window_visible(window)
    game.set_ticrate(ticrate)
    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    return game

def player_action(game, p):
    # Get state variables
    state = game.get_state()

    # Get the buffers
    label_buffer = state.labels_buffer
    depth_buffer = state.depth_buffer

    # Get shooting network inputs
    inputs_shooting = process_img_shooting(label_buffer)

    # Manually create hierarchy in order of importance: Shooting, item gathering, walking
    # If there is an input for the shooting network else check item gathering
    if (np.amax(inputs_shooting, None, None, np._NoValue)) == 1:
        outputs = shooting_net.activate(inputs_shooting)

        # Insert 0 so that outputs correctly represent the action to be took
        outputs.insert(2, 0)

        outputs_from = 'SHOOTING'

        # Checking whether the output will cause an action, if no action revert to walking action
        if (xor((np.amax(outputs, None, None, np._NoValue) == 0), (xor(outputs[0] != 0, outputs[1] != 0)))):
            inputs_walking = process_img_traverse(depth_buffer)
            outputs = walking_net.activate(inputs_walking)
            outputs.append(0)

            outputs_from = 'SHOOTING/WALKING'
    else:
        # Get item gathering inputs
        inputs_item = process_img_item_gather(label_buffer)
        
        # Check if there is any input to the item gathering network if not move
        if (np.amax(inputs_item, None, None, np._NoValue)) == 1:
            # Get output from item network
            outputs = item_net.activate(inputs_item)

            outputs_from = 'ITEM'

            # Check if an action is being made if none, revert to moving actions
            if (np.amax(outputs, None, None, np._NoValue)) == 0:
                inputs_walking = process_img_traverse(depth_buffer)
                outputs = walking_net.activate(inputs_walking)

                outputs_from = 'ITEM/WALKING'

            # Append 0 to end of outputs so that the action is properly represented
            outputs.append(0)

        # Walking actions
        else:
            # Get input for walking network and get output
            inputs_walking = process_img_traverse(depth_buffer)
            outputs = walking_net.activate(inputs_walking)
            outputs.append(0)

            outputs_from = 'WALKING'

    # Make action
    game.make_action(outputs)

    # Display the action that the agent took
    print('PLAYER' + str(p) + ' ' + outputs_from)

    # Respawn player if dead
    if game.is_player_dead():
        game.respawn_player()

def player_host(p):
    game = setup_player()
    game.add_game_args("-host " + str(p) + " -netmode 0 -deathmatch +timelimit " + str(timelimit) +
                       " +sv_spawnfarthest 1 +name Player0 +colorset 0")
    game.add_game_args(args)

    game.init()

    action_count = 0

    for i in range(episodes):
        print("Episode #" + str(i + 1))
        episode_start_time = None

        while not game.is_episode_finished():
            if episode_start_time is None:
                episode_start_time = time()

            state = game.get_state()
            #print("Player0:", state.number, action_count, game.get_episode_time())

            player_action(game, p)
            action_count += 1

        #print("Player0 frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

        print("Host: Episode finished!")
        player_count = int(game.get_game_variable(GameVariable.PLAYER_COUNT))
        for i in range(1, player_count + 1):
            print("Host: Player" + str(i) + ":", game.get_game_variable(eval("GameVariable.PLAYER" + str(i) + "_FRAGCOUNT")))
        print("Host: Episode processing time:", time() - episode_start_time)

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        game.new_episode()

    game.close()


def player_join(p):
    game = setup_player()
    game.add_game_args("-join 127.0.0.1 +name Player" + str(p) + " +colorset " + str(p))
    game.add_game_args(args)

    game.init()

    action_count = 0

    for i in range(episodes):

        while not game.is_episode_finished():
            state = game.get_state()
            #print("Player" + str(p) + ":", state.number, action_count, game.get_episode_time())
            player_action(game, p)
            action_count += 1

        #print("Player" + str(p) + " frags:", game.get_game_variable(GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()

# Load the given filename
def load_genome(filename):
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj

# Create configuration files to recreate the networks
config_walking = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet,
                    stagnation.DefaultStagnation, "../config files/config_walking_V2")
config_item = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet,
                    stagnation.DefaultStagnation, "../config files/config_item_gathering")
config_shooting = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet,
                    stagnation.DefaultStagnation, "../config files/config_shooting_pt2")

# Load in the networks
walking_genome = load_genome("../genomes/moving-genome-PT2.2")
item_genome = load_genome("../genomes/item-gathering-genome")
shooting_genome = load_genome("../genomes/shooting-genome")

# Create the neural networks
walking_net = nn.FeedForwardNetwork.create(walking_genome, config_walking)
shooting_net = nn.FeedForwardNetwork.create(shooting_genome, config_shooting)
item_net = nn.FeedForwardNetwork.create(item_genome, config_item)

if __name__ == '__main__':
    print("Players:", players)
    print("CPUS:", cpu_count())

    processes = []
    for i in range(1, players):
        p_join = Process(target=player_join, args=(i,))
        p_join.start()
        processes.append(p_join)

    player_host(players)

    print("Done")