OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
ry(-2.9953) q[0];
x q[1];
ry(-1.0566) q[1];
cx q[0], q[1];
rz(-1.7144) q[0];
ry(1.1783) q[2];
cx q[1], q[2];
rz(1.4148) q[1];
cx q[0], q[1];
ry(0.5124) q[3];
cx q[2], q[3];
rz(-3.0777) q[2];
cx q[1], q[2];
rz(-1.8281) q[3];
cx q[2], q[3];
