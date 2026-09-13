OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(-2.3599) q[0];
ry(2.1607) q[1];
ry(2.0847) q[2];
ry(0.9114) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(-1.3104) q[0];
rz(-0.9242) q[1];
rz(2.6461) q[2];
rz(0.6138) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
