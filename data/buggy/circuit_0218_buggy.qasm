OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
x q[3];
x q[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
