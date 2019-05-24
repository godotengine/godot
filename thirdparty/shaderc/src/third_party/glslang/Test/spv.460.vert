#version 460

void main()
{
    int a = gl_BaseVertex + gl_BaseInstance + gl_DrawID;
}
