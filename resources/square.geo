dx = 1.; //+ Rough cell size if no argument is provided.
H = 2; //+ Half-Height of domain
W = 1.; //+ Width of Domain

Point(1) = {0, -H, 0, dx};
Point(2) = {W, -H, 0, dx};
Point(3) = {W, +H, 0, dx};
Point(4) = {0, +H, 0, dx};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

Physical Curve("p", 1) = {4};
Physical Curve("a", 2) = {1};
Physical Curve("t", 3) = {2};
Physical Curve("o", 4) = {3};

Physical Surface("domain", 1) = {1};
