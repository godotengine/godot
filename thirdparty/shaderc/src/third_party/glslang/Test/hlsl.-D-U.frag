
#define IN_SHADER

static float4 color;

void main()
{
#if FOO==200
    color = 1.0;
#else
    #error expected FOO 200
#endif

#ifdef FOO
    color -= 5.0;
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
}
