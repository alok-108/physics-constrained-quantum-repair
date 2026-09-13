OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(2.4416) q[0];
ry(0.7196) q[1];
ry(-0.3657) q[2];
ry(1.7137) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(0.1468) q[0];
rz(-1.0802) q[1];
rz(1.489) q[2];
rz(0.0442) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
