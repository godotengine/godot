#line 300

#line 2





#line __LINE__ + 3


#line __FILE__ + 2

#line __FILE__ * __LINE__


#define X 4

#line X

#undef X

#define X(y) y + 3 + 2

#line X(3)

void main() {
  gl_Position = vec4(__LINE__);
}

#line X(3) 4

#define Z(y, q) \
  y*q*2 q

#line Z(2, 3)

#line 1

