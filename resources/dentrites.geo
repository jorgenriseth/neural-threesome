// Gmsh project created on Thu Jun 30 11:08:33 2022
SetFactory("OpenCASCADE");
//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {1, -1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {-1, 1, -0, 1.0};
//+
Point(5) = {-1, 0.3, -0, 1.0};
//+
Point(6) = {-1, -0.3, 0, 1.0};
//+
Point(7) = {-0.25, 0.3, 0, 1.0};
//+
Point(8) = {-0.25, -0.3, -0, 1.0};
//+
Point(9) = {0.25, 0.3, 0, 1.0};
//+
Point(10) = {0.25, -0.3, -0, 1.0};
//+
Point(11) = {1, 0.3, -0, 1.0};
//+
Point(12) = {1, -0.3, 0, 1.0};
//+
Line(1) = {1, 2};
Line(2) = {2, 12};
Line(3) = {12, 10};
Line(4) = {10, 9};
Line(5) = {9, 11};
Line(6) = {11, 3};
Line(7) = {3, 4};
Line(8) = {4, 5};
Line(9) = {5, 7};
Line(10) = {7, 8};
Line(11) = {8, 6};
Line(12) = {6, 1};
Line(13) = {5, 6};
Line(14) = {12, 11};

Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
Plane Surface(1) = {1};

Curve Loop(2) = {11, -13, 9, 10};
Plane Surface(2) = {2};

Curve Loop(3) = {3, 4, 5, -14};
Plane Surface(3) = {3};

Physical Curve("ecs_outer_boundary", 1) = {1, 2, 6, 7, 8, 12};
Physical Curve("axon_outer_boundary", 2) = {13};
Physical Curve("axon_ecs_membrane", 3) = {9, 11};
Physical Curve("axon_synapse_membrane", 4) = {10};
Physical Curve("dendrite_ecs_membrane", 5) = {5, 3};
Physical Curve("dendrite_synapse_membrane", 6) = {4};
Physical Curve("dendrite_outer_boundary", 7) = {14};

Physical Surface("ecs", 1) = {1};
Physical Surface("axon", 2) = {2};
Physical Surface("dendrite", 3) = {3};
