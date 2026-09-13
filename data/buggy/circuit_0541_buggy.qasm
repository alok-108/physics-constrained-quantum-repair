OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q
x q[0];
x q[1];
ry(0.2386) q[0];
ry(-2.4223) q[1];
ry(1.2388) q[2];
ry(0.4156) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
rz(0.6612) q[0];
rz(-1.9181) q[1];
rz(0.921) q[2];
rz(1.3633) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
