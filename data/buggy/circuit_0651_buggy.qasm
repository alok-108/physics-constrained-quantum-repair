OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
ry(-0.4795) q[0];
x q[1];
ry(1.4738) q[1];
cx q[0], q[1];
rz(-2.7892) q[0];
ry(-1.6685) q[2];
cx q[1], q[2];
rz(2.058) q[1];
cx q[0], q[1];
ry(1.4059) q[3];
cx q[2], q[3];
rz(2.1939) q[2];
cx q[1], q[2];
rz(2.4842) q[3];
cx q[2], q[3];
