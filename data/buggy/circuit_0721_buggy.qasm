OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
ry(-1.1872) q[0];
x q[1];
ry(-0.1102) q[1];
cx q[0], q[1];
rz(1.8717) q[0];
ry(0.0009) q[2];
cx q[1], q[2];
rz(-0.8847) q[1];
cx q[0], q[1];
ry(-2.9983) q[3];
cx q[2], q[3];
rz(-0.7404) q[2];
cx q[1], q[2];
rz(-2.483) q[3];
cx q[2], q[3];
