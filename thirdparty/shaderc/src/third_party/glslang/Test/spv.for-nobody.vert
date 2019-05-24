#version 450
layout(location=0) out highp int r;
void main() {
  int i;
  for (i=0; i<10; i++);
  r = i;
}
