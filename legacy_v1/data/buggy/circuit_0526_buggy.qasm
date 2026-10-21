OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(-2.5007) q[0];
ry(0.2355) q[1];
ry(2.0312) q[2];
ry(3.096) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(-1.4298) q[0];
rz(-1.8551) q[1];
rz(1.4262) q[2];
rz(-0.3718) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
