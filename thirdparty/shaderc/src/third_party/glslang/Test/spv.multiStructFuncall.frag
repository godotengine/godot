#version 450

struct S { mat4 m; };
buffer blockName { S s1; };  // need an S with decoration
S s2;                        // no decorations on S

void fooConst(const in S s) { }
void foo(in S s) { }
void fooOut(inout S s) { }

void main()
{
  fooConst(s1);
  fooConst(s2);

  foo(s1);
  foo(s2);

  fooOut(s1);
  fooOut(s2);
}