SetFactory("OpenCascade");

res = 50;

width_cleft = 1.0;
domain_length = 3.0;
domain_height = 2.0;

terminal_height = 1.8;

grid_size = width_cleft / res;

// pre-synaptic side
Point(1) = {0, -domain_height/2, 0, grid_size};
Point(2) = {0, -terminal_height/2, 0, grid_size};
Point(3) = {0, terminal_height/2, 0, grid_size};
Point(4) = {0, domain_height/2, 0, grid_size};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Physical Curve("Pre-synaptic Terminal", 2) = {2};

// top
Point(5) = {domain_length, domain_height/2, 0, grid_size};
Line(4) = {4, 5};

// post-synaptic side
Point(6) = {domain_length, terminal_height/2, 0, grid_size};
Point(7) = {width_cleft, terminal_height/2, 0, grid_size};
Point(8) = {width_cleft, -terminal_height/2, 0, grid_size};
Point(9) = {domain_length, -terminal_height/2, 0, grid_size};
Point(10) = {domain_length, -domain_height/2, 0, grid_size};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Physical Curve("Post-synaptic Terminal", 3) = {7};
Physical Curve("Dendrite Membrane", 5) = {6, 8};

// bottom
Line(10) = {10, 1};
Physical Curve("Astrocyte", 4) = {10};
Physical Curve("Boundary", 1) = {1, 3, 4, 5, 9};

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
Plane Surface(1) = {1};

Physical Surface("Extracellular Domain", 1) = {1};

Mesh.Algorithm = 6;
