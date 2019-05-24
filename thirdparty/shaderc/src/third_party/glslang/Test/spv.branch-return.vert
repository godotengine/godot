#version 310 es
void main() {
  switch (gl_InstanceIndex) {
    case 0: return;
    case 1: gl_Position = vec4(0.0); break;
    case 2: return;
    case 3: return;
  }
  gl_Position.x += 0.123;
}
