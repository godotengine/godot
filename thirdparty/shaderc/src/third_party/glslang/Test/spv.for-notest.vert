#version 450
layout(location=0) out highp int r;
void main() {
  int i;
  for (i=0; ; i++) { r = i; }
}
