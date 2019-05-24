#version 450
layout(location=0) out highp int r;
layout(location=0) in lowp int flag;
void main() {
  int i;
  for (i=0; i < (flag==1 ? 10 : 15) ; i++) { r = i; }
}
