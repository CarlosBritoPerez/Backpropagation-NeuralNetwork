

def getColors(predictions):

    for x in range(len(predictions)):
        if predictions[x] < 0.2:
            print('%.2f ---> Green' % predictions[x])
        elif predictions[x] < 0.4:
            print('%.2f ---> Lightgreen' % predictions[x])
        elif predictions[x] < 0.6:
            print('%.2f ---> Yellow' % predictions[x])
        elif predictions[x] < 0.8:
            print('%.2f ---> Orange' % predictions[x])
        elif predictions[x] <= 1:
            print('%.2f ---> Rojo' % predictions[x])


