#version 440

uniform sampler smp;
uniform texture2D tex;

in vec2 coord;

out vec4 color;

void main()
{
  color = texture(sampler2D(tex, smp), coord);
}
