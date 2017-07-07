import sys
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))
from util.readData import getColumn
import json

businessAreaColumn = 'D'

distinctAreas = {}
areaID = 0
for item in getColumn(businessAreaColumn):
    if item not in distinctAreas.values():
        distinctAreas['{:03d}'.format(areaID)] = item
        areaID += 1

outputFile = open('getBusinessAreas.json', 'w')
outputFile.write(json.dumps(distinctAreas))
outputFile.close()