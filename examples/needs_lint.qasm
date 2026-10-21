OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
h q[0];
x q[1];
x q[1];
cx q[0], q[3];
rz(0) q[2];
