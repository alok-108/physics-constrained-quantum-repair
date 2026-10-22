OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
for int k in [0:0] {
  rz(0.5774) q[0];
  cx q[0], q[1];
  rx(0.7324) q[2];
}
cx q[1], q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
