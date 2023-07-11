dx = 0.1; //+ Rough cell size if no argument is provided.
H = 0.5; //+ Half-Height of domain
W = 0.5; //+ Half-Width of Domain

Point(1) = {-W, -H, 0, dx};
Point(2) = {+0, -H, 0, dx};
Point(3) = {+W, -H, 0, dx};
Point(4) = {+W, +H, 0, dx};
Point(5) = {+0, +H, 0, dx};
Point(6) = {-W, +H, 0, dx};
Point(7) = {+0, +0, 0, dx};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};
Line(7) = {2, 7};
Line(8) = {7, 5};

Curve Loop(1) = {1, 7, 8, 5, 6};
Plane Surface(1) = {1};

Curve Loop(2) = {2, 3, 4, -8, -7};
Plane Surface(2) = {2};

Physical Curve("left-boundary", 1) = {6};
Physical Curve("right-boundary", 2) = {3};
Physical Curve("interface-bottom", 3) = {7};
Physical Curve("interface-top", 4) = {8};

Physical Surface("e", 1) = {1};
Physical Surface("i", 2) = {2};

