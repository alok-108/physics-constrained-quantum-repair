OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(-2.6924) q[0];
ry(2.8253) q[1];
ry(-1.227) q[2];
ry(-0.7211) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(-1.0585) q[0];
rz(-1.0429) q[1];
rz(-0.5541) q[2];
rz(2.8574) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
