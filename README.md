## Comp Phys Project, group G

**Project title:** 
Something like "Daydreaming on Hopfield and Potts neural networks" or "Physics inspired biological neural network".

**Project Description:**
This project explores how ideas from statistical physics, specifically the Ising model, have inspired the development of associative neural networks such as the Hopfield Neural Network (HNN) and the Potts Neural Network (PNN). Our project aims to connect physical models of interacting spins to computational models of memory and learning. We will demonstrate the analogy between physical systems and neural computation by evolving the Ising model, HNN, and PNN using the Metropolis algorithm and analyzing their energy landscapes, critical phenomena, and pattern retrieval. We will then incorporate a recent “Daydreaming” learning rule (as described in https://www.sciencedirect.com/science/article/pii/S0893608025000954) to examine how it enhances memory capacity, improves pattern retrieval, and changes the energy landscape. 
   
**The (loosely) planned directory structure is as follows:** 
<pre> 
    /README.md 
    /Ising.ipynb     #simulation of the 2D Ising model using metropolis alogorithm
    /HNN.ipynb       #implementation of the HNN, with Hebbian and Daydreaming learning rules 
    /viz.py          #functions for visualizing/plotting spin configurations, energy landscape, and memory retrieval
    /ref_data.py     #reference data for test/train, either imported or hardcoded
    /results.ipynb   #jupyter notebook displaying overall results 
</pre>

**Chronological steps to complete the project:** 
- Ising Model Implementation:
   - Simulate the 2D Ising model using the Metropolis algorithm.
   - Visualize spin configurations and compute magnetization vs. temperature to observe the critical phase transition.
   - Explore different geometric configurations.
- Hopfield Network (2-state Ising analogue):
   - Implement a HNN using the Hebbian learning rule.
   - Study/Visualize memory retrieval dynamics and explore energy landscape convergence.
   - Incorporate the Daydreaming learning rule to reduce spurious attractors and enhance memory capacity.


**Planned contributions of each team member:**
Nancy: Implement base square ising model, implement daydreaming learning method on hopfield network
Sam: Implement triangular ising model, implement base hopfield network
Shivam: Implement hexagonal ising model, implement visulaization classes

**Additional notes:**
If time permits, we may also implement the Potts model with and without the daydreaming algorithm.     
    
