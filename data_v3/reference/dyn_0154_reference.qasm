OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
ry(0.6611) q[0];
h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];
c[0] = measure q[0];
c[1] = measure q[1];
if (c[1] == true) {
  x q[2];
}
if (c[0] == true) {
  z q[2];
}
c[2] = measure q[2];
