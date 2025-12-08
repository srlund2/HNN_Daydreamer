## Comp Phys Project, group G

**Ising on cake (and other Hopfield daydreams)** 
An analysis of the simplfied Ising model on square, triangular, and hexagonal lattices without an external field. A square lattice implementaiton with an external field is also used. We demonstrate the analogy between physical systems and neural computation by evolving the Ising model using the Metropolis algorithm, thus inducing a noise-type temperature dependence, and analyzing the energy landscapes and critical phenomena.

The simplified Ising model is defined by the Hamiltonian:
$E = -J \sum_(<i, j>) s_i s_j - H \sum_i s_i$

Additionally, the 2D analog of the Ising model---the hopfield nueral network (HNN)---is implemented and tested on handwritten digits and salt and pepper images. The "daydreaming" algorithm (https://www.sciencedirect.com/science/article/pii/S0893608025000954)---a learning rule to increase memory capacity---is implemented. We compare this daydreaming algorithm's pattern retrieval abilities with a standard Hebbian rule HNN.

   
**The directory structure is as follows:** 
<pre> 
    /README.md 
    /Ising_with_numpy.ipynb     #simulation of the 2D Ising model using metropolis alogorithm. Implemented in numpy, on square, triangular, and hexagonal lattices. 
    /Ising_with_networkx.ipynb       #simulation of the 2D Ising model implemented using the networkx and graphviz packages, for improved visualization.
    /Ising_in_external_field.ipynb     #simulation of the 2D Ising model on a square lattice, with a time and position dependent external magnetic field.
    /hopfield_daydreaming.ipynb     #implementation of the HNN, with Hebbian and Daydreaming learning rules 
</pre>
Each file is a jupyter notebook, to provide maximum flexibility for users. Further details on the contents of each file can be found in a markdown box at the begining of each. 

**Signifigant results:**
We found that our simmulated Ising model phase transitions agreed with the theoretical predictions for the value of $T_C$, to within 0.2 T (J/K). More intensive simulations could be run by increasing the runs averaged over, the number of steps per run, and the lattice size of the simulation.\
We also found that with an external field, the square Ising model behaves as expected, following the magnetic field spatially or temporally, with some noise and lag.\  
A presentation detialing the major fidnings and basic theory of the implementation can be found at https://docs.google.com/presentation/d/1TGY32IlcQUvz-I5vmVRkyGaN0YJjL7VBd71pUhsn8ig/edit?usp=sharing. 

**Future iterations:**
To expand this project, the hopfield neural net could be further tested with more complicated images, to see how HNN performs on more or less correlated data when compared with the daydreaming learning rule. Other neural nets, such as the Potts neural net, could aslo be implement with and without the daydreaming learning rule.
    
