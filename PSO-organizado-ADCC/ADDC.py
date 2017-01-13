'''
Created on 25/08/2016

@author: aliss
'''

class ADDC():
    def __init__(self):
        pass
    def Metric_Calculate(self, dissimi_matrix, centro_vector, doc_b_cluster):
        
        dissimilarity_matrix = dissimi_matrix #dissimilarity matrix of documents
        centroid_vector = centro_vector           #Vector containing the indices of each centroid in the dissimilarity matrix.
        documents_b_cluster = doc_b_cluster    #Matrix lists, where each list contains the indices of each pertecente document to each centroid

        dissimilarity_sum = 0
        
        
        for i in range(len(centroid_vector)):
            
            dissimilarity_sum_per_cluster = 0
            
            for j in range(len(documents_b_cluster[0]) - 1):

               dissimilarity_sum_per_cluster += dissimilarity_matrix[centroid_vector[i]-1][documents_b_cluster[i][j][0]]

            dissimilarity_sum_per_cluster = dissimilarity_sum_per_cluster / len(documents_b_cluster[i])
            
            dissimilarity_sum = dissimilarity_sum + dissimilarity_sum_per_cluster
            print(dissimilarity_sum)
        
        dissimilarity_sum = dissimilarity_sum / len(centroid_vector)
        
        return dissimilarity_sum    