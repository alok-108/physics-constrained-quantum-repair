OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(-2.8622) q[0];
ry(3.0804) q[1];
ry(2.122) q[2];
ry(-0.9755) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(-2.6489) q[0];
rz(-0.8512) q[1];
rz(1.7831) q[2];
rz(-2.0356) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
