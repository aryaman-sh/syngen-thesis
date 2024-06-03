# Parameters needed for lab2im generation 

generation_labels: Strunctures from which to generate. i.e. all discrete labels in the image 
    eg. [0, 14, 15, 16, 24, 72, 85, 165, 258, 259, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 30, 31]
output_labels: Structures we want to keep
    eg. [0, 14, 15, 16, 0, 0, 0, 0, 0, 0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 0, 31]
generation_classes: regrouping clusters into classes so they share the same distribution
prior_means: k (number of classes) means
prior_std: number of classes standard deviations