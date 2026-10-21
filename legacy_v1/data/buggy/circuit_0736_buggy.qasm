OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(-2.8743) q[0];
ry(-2.3344) q[1];
ry(2.6403) q[2];
ry(2.0731) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(2.4958) q[0];
rz(2.0935) q[1];
rz(-0.5632) q[2];
rz(-1.2409) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
