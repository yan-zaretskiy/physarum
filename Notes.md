## sin and cos implementations:
### Setup/Info:
- measured in ms/agent ticked
- 2048 iterations
- average of 3 trials
- 256x256 grid
- 1 << 20 particles
- 1 population
### Results:
- normal sin + normal cos: 
    - 0.000018192ms
- old sin + old cos:
    - 0.000019803ms (8.85% slower)
- fast_approx::fast::sin + fast_approx::fast::cos
    - 0.000018658ms (2.56% slower)
- fast_approx::faster::sin + fast_approx::faster::cos
    - 0.000015878ms (14.57% faster)