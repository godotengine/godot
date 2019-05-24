#version 400

layout(push_constant) uniform Material {
    int kind;
    float fa[3];
} matInst;

out vec4 color;

void main()
{
    switch (matInst.kind) {
    case 1:  color = vec4(0.2); break;
    case 2:  color = vec4(0.5); break;
    default: color = vec4(0.0); break;
    }
}
