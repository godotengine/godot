#version 310 es
void main() {
  int i;
  int A, B, C, D, E, F, G;
  for (i=0; i < 10 ; i++) {
    A = 1;
    if (i%2 ==0) {
      B = 1;
      continue;
      C = 1;
    }
    if (i%3 == 0) {
      D = 1;
      break;
      E = 1;
    }
    F = 12;
  }
  G = 99;
}
