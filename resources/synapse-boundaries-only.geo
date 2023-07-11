dx = 0.1; //+ Rough cell size if no argument is provided.
H = 1.0; //+ Distance from center to top of domain
W = 2.0; //+ Total width of domain 
synapse_length = 1.;  //+ Length of synaptic cleft
synapse_width = 0.9 * 2. * H; //+ Width of synaptic cleft
astrocyte_start = 0.1; 
astrocyte_end = 1.5;

Point(1) = {0, -H, 0, dx};
Point(2) = {astrocyte_start, -H, 0, dx};
Point(3) = {astrocyte_end, -H, 0, dx};
Point(4) = {W, -H, 0, dx};
Point(5) = {W, -synapse_width/2, 0,  dx};
Point(6) = {synapse_length, -synapse_width/2, 0, dx};
Point(7) = {synapse_length, +synapse_width/2, 0, dx};
Point(8) = {W, +synapse_width/2, 0, dx};
Point(9) = {W, H, 0, dx};
Point(10) = {0, H, 0, dx};
Point(11) = {0, synapse_width/2, 0, dx};
Point(12) = {0, -synapse_width/2, 0, dx};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
Plane Surface(1) = {1};

Physical Curve("pre", 1) = {11};
Physical Curve("astrocyte", 2) = {2};
//+Physical Curve("astrocyte", 2) = {1, 2, 3}; //+ Uncomment if we want entire bottom as astrocyte
Physical Curve("post-side", 3) = {5, 7};
Physical Curve("post", 4) = {6};

Physical Surface("domain", 1) = {1};
