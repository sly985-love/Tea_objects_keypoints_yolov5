import torch


s = '0, 0.43431712962962965, 0.2981770833333333, 0.07966820987654322, 0.30063657407407407, 0.466820987654321, 0.2175925925925926, 0.4498456790123457, 0.33969907407407407, 0.40335648148148145, 0.16145833333333334, 0.4419367283950617, 0.33564814814814814, 0.4427083333333333, 0.3732638888888889, 0.42939814814814814, 0.4363425925925926, 0, 0.6059992283950617, 0.26461226851851855, 0.05999228395061729, 0.23582175925925927, 0.5814043209876543, 0.2230902777777778, 0.5848765432098766, 0.30497685185185186, 0.6242283950617284, 0.15335648148148148, 0.5939429012345679, 0.2902199074074074, 0.5895061728395061, 0.32349537037037035, 0.5875771604938271, 0.3758680555555556, 0, 0.9637345679012346, 0.23451967592592593, 0.07253086419753087, 0.2803819444444444, 0.9587191358024691, 0.09953703703703703, 0.9373070987654321, 0.27141203703703703, 1.001543209876543, 0.21354166666666666, 0.9506172839506173, 0.28877314814814814, 0.9373070987654321, 0.3142361111111111, 0.9346064814814815, 0.36516203703703703'
a = torch.tensor(((0, 0.43431712962962965, 0.2981770833333333, 0.07966820987654322, 0.30063657407407407, 0.466820987654321, 0.2175925925925926, 0.4498456790123457, 0.33969907407407407, 0.40335648148148145, 0.16145833333333334, 0.4419367283950617, 0.33564814814814814, 0.4427083333333333, 0.3732638888888889, 0.42939814814814814, 0.4363425925925926), (0, 0.6059992283950617, 0.26461226851851855, 0.05999228395061729, 0.23582175925925927, 0.5814043209876543, 0.2230902777777778, 0.5848765432098766, 0.30497685185185186, 0.6242283950617284, 0.15335648148148148, 0.5939429012345679, 0.2902199074074074, 0.5895061728395061, 0.32349537037037035, 0.5875771604938271, 0.3758680555555556),( 0, 0.9637345679012346, 0.23451967592592593, 0.07253086419753087, 0.2803819444444444, 0.9587191358024691, 0.09953703703703703, 0.9373070987654321, 0.27141203703703703, 1.001543209876543, 0.21354166666666666, 0.9506172839506173, 0.28877314814814814, 0.9373070987654321, 0.3142361111111111, 0.9346064814814815, 0.36516203703703703)),)

a = a[:, 1:]
print(a)
for j in range(len(a)):
    for i in range(len(a[0])):
        a[j][i] = a[j][i] * 640
print(a)