# Complete generation pipeline

### Generation steps
1. Clustering 
2. label generation / prior estimation
3. Sampling parameters using mask

Given Images and labels directory synthetic images.  
### Generation modes
1. Single sample based generation: Clustering and prior estimation is done sample wise
2. Dataset wide generation: Priors are sampled from the complete dataset.




### Progress marker
[X] Given diretory, gives different clusters for diffferent images  
    - NOTE: Clusters are not consistent across images   
[X] Can generate all the labels   
[ ] Prior estimation   
    -NOTE: Currently performing sample wise prior estimation   
