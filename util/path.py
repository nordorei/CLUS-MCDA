import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir)), os.pardir))
FEED_DATA_PATH = ROOT_PATH + '/data/feed-data.xlsx'
BUSINESS_AREAS_PATH = ROOT_PATH + '/data/getBusinessAreas.json'