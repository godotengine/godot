#version 310 es
void main() {
  int i = 0;
  int A, B, C, D, E, F, G;
  do {
    A = 0;
    if (i == 2) {
      B = 1;
      continue;
      C = 2;
    }
    if (i == 5) {
      D = 3;
      break;
      E = 42;
    }
    F = 99;
  }  while (++i < 19);
  G = 12;
}
