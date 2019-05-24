#version 310 es
#define X(Y) /*
                */ Y + 2

#define Y(Z) 2 * Z// asdf

#define Z(Y) /*
                */ \
  2 /*
       */ + 3 \
    * Y

void main() {
  gl_Position = vec4(X(3) + Y(4) + Z(2));
}
