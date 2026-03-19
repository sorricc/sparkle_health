import os
from openai import AzureOpenAI
import pandas as pd
from sqlalchemy import create_engine
import json
import re
from tabulate import tabulate
from collections import defaultdict, Counter
import time
import random
from dotenv import load_dotenv


load_dotenv() 

API_KEY = os.getenv("API_KEY_PRD")
API_VERSION = '2024-08-01-preview'
deployment = 'gpt-4o-2024-08-06'
RESOURCE_ENDPOINT = os.getenv("RESOURCE_ENDPOINT_PRD")
conn = os.getenv("SCUBA_PRD")
conn2 = os.getenv("SCUBA_DEV")