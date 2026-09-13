OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(2.6307) q[0];
ry(-0.3737) q[1];
ry(0.7715) q[2];
ry(2.8138) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(0.8419) q[0];
rz(1.2052) q[1];
rz(-0.9766) q[2];
rz(-1.0742) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
