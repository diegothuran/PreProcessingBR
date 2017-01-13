import csv
import numpy as np

def readBase(csvFile = str, header = 1):
    dictionary = {"Positivo":0,'"Positivo"':0,"Negativo":1, '"Negativo"':1, "Neutro":2}
    saidas = []
    with open(csvFile, 'rb') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in spamreader:
            print row
            try:
                if header == 0:
                    pass
                    #when the document has header
                else:
                    saidas.append(dictionary[row[0]])
                header += 1
            except IndexError:
                print row
                pass
        return saidas

if __name__ == '__main__':
    saida = readBase('../dataset/unbalanced_corpus/unbalanced_two_class_saida.csv')

    np.savetxt('unbalanced_two_class_saida.txt', saida)