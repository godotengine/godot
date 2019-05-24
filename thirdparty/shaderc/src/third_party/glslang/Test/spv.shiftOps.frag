#version 450

flat in int   i1;
flat in uint  u1;
flat in ivec3 i3;
flat in uvec3 u3;

out ivec3 icolor;
out uvec3 ucolor;

void main()
{
    icolor = i3 << u1;
    icolor <<= 4u;

    ucolor = u3 >> i1;
    ucolor >>= 5;
}
