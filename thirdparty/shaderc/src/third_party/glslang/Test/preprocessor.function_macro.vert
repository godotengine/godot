#version 310 es


#define X(n) n + 1
#define Y(n, z) n + z
#define Z(f) X(f)

#define REALLY_LONG_MACRO_NAME_WITH_MANY_PARAMETERS(X1, X2, X3, X4, X5, X6, X7,\
    X8, X9, X10, X11, X12) X1+X2+X3+X4+X5+X6+X7+X8+X9+X10+X11+X12

#define A(\
  Y\
  )\
4 + 3 + Y

int main() {
  gl_Position = vec4(X(3), Y(3, 4), Z(3));
  gl_Position = vec4(REALLY_LONG_MACRO_NAME_WITH_MANY_PARAMETERS(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
  gl_Position = vec4(A(3));
}
