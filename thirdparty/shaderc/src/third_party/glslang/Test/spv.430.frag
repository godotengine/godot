#version 430 core

out vec4 color;

void main()
{
     color = vec4(1.0);
     color *= gl_Layer;
     color *= gl_ViewportIndex;
}
