OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[0];
cx q[1], q[2];
