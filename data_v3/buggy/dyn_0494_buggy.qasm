OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
bit[3] c;
h q[0];
for int k in [0:3] {
  rz(0.5444) q[0];
  cx q[0], q[1];
  rx(0.5974) q[2];
}
cx q[2], q[1];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
