#version 310 es
#define X 1
#define Y clamp
#define Z X

#define F 1, 2

#define make_function \
  float fn ( float x ) \
  {\
    return x + 4.0; \
  }

make_function

int main() {
  gl_Position = vec4(X);
  gl_Position = Y(1, 2, 3);
  gl_Position = vec4(Z);
  gl_Position = vec4(F);
  gl_Position = vec4(fn(3));
  [] . ++ --
  + - * % / - ! ~
  << >> < > <= >=
  == !=
  & ^ | && ^^ || ? :
  += -= *= /= %= <<= >>= &= |= ^=
  1.2 2E10 5u -5lf
}

struct S {
    int member1;
    float member2;
    vec4 member3;
};

#define xyz xxyz
#define yzy() yyz

#define FUN_MAC() \
	vec3 a = vec3(0); \
	vec3 b = a.zxyz;  \
	vec3 b = a.xyz;   \
	vec3 b = a.yzy();   \
	vec3 b = a.xyz();   \
	vec3 b = a.yzy;   \
	vec3 b = a.z;

void foo()
{
    S s;
    s.member2 + s.member1;
    s.member3.zyx;
    s.member2.xyz;
    s.member2.yzy();
    s.member2.xyz();
    s.member2.yzy;
    FUN_MAC()
    yzy

    ();
    yzy


}
