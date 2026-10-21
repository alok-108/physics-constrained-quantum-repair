OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(2.857) q[0];
ry(1.8773) q[1];
ry(0.0352) q[2];
ry(-2.2116) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(-2.647) q[0];
rz(-0.6731) q[1];
rz(2.9199) q[2];
rz(0.8037) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
