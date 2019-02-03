import visualize
import pickle
from neat import *

def load_genome(filename):
    file = open(filename, 'rb')
    obj = pickle.load(file)
    file.close()

    return obj

config_walking = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet,
                        stagnation.DefaultStagnation, "../config files/config_walking_V2")
config_health = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet,
                       stagnation.DefaultStagnation, "../config files/config_health_gathering")
config_shooting = Config(genome.DefaultGenome, reproduction.DefaultReproduction, species.DefaultSpeciesSet,
                         stagnation.DefaultStagnation, "../config files/config_shooting_pt2")

walking_genome = load_genome("../genomes/moving-genome-PT2.2")
health_genome = load_genome("../genomes/Health-gathering-genome")
shooting_genome = load_genome("../genomes/shooting-genome")

node_names = {0:"Turn Left", 1:"Turn Right", 2:"Move Forward", -1:"Left 4", -2:"Left 3", -3:"Left 2", -4:"Left 1", -5:"Centre 1",
              -6:"Centre 2", -7:"Centre 3", -8:"Centre 4", -9:"Centre 5", -10:"Centre 6", -11:"Right 1", -12:"Right 2", -13:"Right 3", -14:"Right 4"}
visualize.draw_net(config_health, health_genome, True, node_names=None, show_disabled=False)