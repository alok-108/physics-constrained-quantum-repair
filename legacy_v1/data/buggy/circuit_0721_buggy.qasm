OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(0.7591) q[0];
ry(2.9551) q[1];
ry(-2.3846) q[2];
ry(-2.8089) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(0.0935) q[0];
rz(1.4937) q[1];
rz(0.1434) q[2];
rz(2.1112) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
