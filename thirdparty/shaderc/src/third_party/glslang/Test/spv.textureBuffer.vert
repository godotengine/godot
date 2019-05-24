#version 450

uniform textureBuffer tBuf;
uniform sampler s;
uniform samplerBuffer sBuf;

uniform utextureBuffer utBuf;
uniform itextureBuffer itBuf;

void main()
{
    texelFetch(samplerBuffer(tBuf, s), 13);
    texelFetch(sBuf, 13);
    texelFetch(tBuf, 13);
    texelFetch(utBuf, 13);
    texelFetch(itBuf, 13);
}
