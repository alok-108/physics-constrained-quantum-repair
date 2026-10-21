OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
ry(-2.311) q[0];
x q[1];
ry(2.0293) q[1];
cx q[0], q[1];
rz(-1.7179) q[0];
ry(-1.9977) q[2];
cx q[1], q[2];
rz(1.6097) q[1];
cx q[0], q[1];
ry(-2.8823) q[3];
cx q[2], q[3];
rz(-2.5297) q[2];
cx q[1], q[2];
rz(2.2471) q[3];
cx q[2], q[3];
