# numerical schemes for integration

### outline of project

## Documentation
### Kepler
the files that have to do with the kepler program are the ones prefixed by 'kep_', these contain code that has to do with the two body problem in two d.

### modified_equations.py

##### format of pishr
Lists with var name pishr store information about polynomials in h in terms of the functions f_i(\tilde y). The format is as follows
[i , ["fk",n]] means $$h^i f_k^{(n)}(\tilde y)$$ where $$f_1 = f$$


