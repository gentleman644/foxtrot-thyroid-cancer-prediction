
#Purpose: find out with the given text if the text is in float format
#Parameter: string
#Result: True or False
#Author: Tim Liu
def float_verification(text: str):
    list = text.split('.')

    if len(list) == 1 and (list[0].isdigit() or list[0] == ''):
        return True
    elif len(list) == 2 and (list[0].isdigit() or list[0] == '') and (list[1].isdigit() or list[1] == ''):
        return True
    else:
        return False
