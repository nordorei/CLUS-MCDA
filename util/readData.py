import openpyxl as xlsx

workbook = xlsx.load_workbook(filename='../data/feed-data.xlsx')
sheet = workbook.get_sheet_by_name(name='Big Data Structured Matrix')

def getColumn(column):
    """
    """
    result = []
    for row in range(2, sheet.max_row):
        cellName = "{}{}".format(column, row)
        result.append(sheet[cellName].value)

    return result

