OPENQASM 3.0;
include "stdgates.inc";
qbit_invalid[3] q;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
