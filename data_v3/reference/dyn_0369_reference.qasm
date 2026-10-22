OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
for int k in [0:1] {
  rz(0.7781) q[0];
  cx q[0], q[1];
  rx(1.0468) q[2];
}
cx q[1], q[2];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
