#main file of the imbalancedClass project

#importing packages

import argeparse
import logging

#Setting up log file and messages we want into the log file
logging.basicConfig(filename='Log_File.log', encoding='utf-8', level=logging.DEBUG)


#Setting up arguments for the file and writing help for the main file 
parser = argparse.ArgumentParser()
parser.add_argument("classification_model", help="specify the algorithm for classification",
                    type=string)