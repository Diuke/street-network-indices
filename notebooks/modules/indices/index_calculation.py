import json
import math
import copy
import scipy
import os
import time
import numpy as np
from pathlib import Path
import pyproj
import networkx as nx
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as geopd
import shapely
from geopy import distance as geopy_distance
from shapely import ops, LineString
import requests
from itertools import chain
from modules import utils
from networkx import bfs_edges
from modules import generalize
import sys

