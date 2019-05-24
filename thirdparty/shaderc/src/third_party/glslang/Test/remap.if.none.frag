#version 450

in float  inf;
out vec4  outf4;

void main()
{
    if (inf > 2.0)
        outf4 = vec4(inf);
    else
        outf4 = vec4(inf + -.5);
}
