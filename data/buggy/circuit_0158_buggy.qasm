OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
x q[2];
x q[2];
cx q[0], q[1];
cx q[1], q[2];
