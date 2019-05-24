#version 450
precision mediump int; precision highp float;
layout(location=1) in highp vec4 v;
void main (void)
{
  if (v == vec4(0.1,0.2,0.3,0.4)) discard;
  else return;
}
