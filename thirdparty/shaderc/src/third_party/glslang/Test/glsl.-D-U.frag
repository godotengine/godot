#version 450

#define IN_SHADER

layout(location=0) out vec4 color;

void main()
{
#if FOO==200
    color = vec4(1.0);
#else
    #error expected FOO 200
#endif

#ifdef IN_SHADER
    color++;
#else
    #error IN_SHADER was undef
#endif

#ifdef UNDEFED
    #error UNDEFED defined
#else
    color *= 3.0;
#endif

#if MUL == 400
    color *= MUL;
#else
    #error bad MUL
#endif
}
