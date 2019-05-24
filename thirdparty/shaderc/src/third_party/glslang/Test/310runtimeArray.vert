#version 310 es

precision highp float;
layout(location=0) out float o;

struct S  { float f; };
buffer b1 { S s[]; };
buffer b2 { S s[]; } b2name;
buffer b3 { S s[]; } b3name[];
buffer b4 { S s[]; } b4name[4];

void main()
{
  o = s[5].f;
  o += b2name.s[6].f;
  o += b3name[3].s[7].f;
  o += b4name[2].s[8].f;
}
