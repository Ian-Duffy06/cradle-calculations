As it stands, the code is set to a HSS 2x2x1/4" beam. 
If you want to change the beam type, you do need to find the ASIC manual for HSS properties and input them in Cell #1 under beam properties.
The only imputs that matter for anyone using this lie within cell #2. 
You can change the CASE to be a point (worse case), or normal (regular distributed load). 
I am currently adding a third case for lifting that takes all of those facotors into account and spits out the max start and stopping speed in both the vertical and horizontal direction.
