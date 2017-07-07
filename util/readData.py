import openpyxl as xlsx
import numpy as np

workbook = xlsx.load_workbook(filename='../data/feed-data.xlsx')
sheet = workbook.get_sheet_by_name(name='Big Data Structured Matrix')

def getColumn(column):
    """
    (String) -> (List)

    Gets the column from Big Data Structured Matrix sheet with the name of '@param:column'
    """
    result = []
    for row in range(3, sheet.max_row):
        cellName = "{}{}".format(column, row)
        result.append(sheet[cellName].value)

    return result

def getSuppliersData():
    """
    """
    suppliers = {}
    for row in range(3, sheet.max_row):
        hasNullData = False
        sCode = sheet['B{}'.format(row)].value
        sDataList = []
        for column in "EFGHIJK":
            cellName = "{}{}".format(column, row)
            cellValue = sheet[cellName].value
            if cellValue is None:
                hasNullData = True
                break
            sDataList.append(cellValue)

        if hasNullData:
            continue

        sDataArray = np.array(sDataList)
        suppliers[sCode] = sDataArray

    return suppliers