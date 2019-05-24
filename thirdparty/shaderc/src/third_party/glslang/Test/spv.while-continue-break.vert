#version 310 es
void main() {
  int i = 0;
  int A, B, C, D;
  while (i<10) {
    A = 1;
    if (i%2 == 0) {
      B = 2;
      continue;
      C = 2;
    }
    if (i%5 == 0) {
      B = 2;
      break;
      C = 2;
    }
    i++;
  }
  D = 3;
}
