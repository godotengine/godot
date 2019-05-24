#version 100

varying vec3 color;       // ERRROR, there is no default qualifier for float

lowp vec2 foo(mediump vec3 mv3)
{
    highp vec4 hv4;
    return hv4.xy;
}

int global_medium;

uniform lowp sampler2D samplerLow;
uniform mediump sampler2D samplerMed;
uniform highp sampler2D samplerHigh;

precision highp int; 
precision highp ivec2;     // ERROR
precision mediump int[2];  // ERROR
vec4 uint;                 // okay
precision mediump vec4;    // ERROR

int global_high;

void main()
{
    lowp int sum = global_medium + global_high;

    gl_FragColor = vec4(color, 1.0);

    int level1_high;
    sum += level1_high;

    precision lowp int;
    int level1_low;
    sum += level1_low;
    
    // test maxing precisions of args to get precision of builtin
    lowp float arg1;
    mediump float arg2;
    lowp float d = distance(arg1, arg2);

    {
        int level2_low;
        sum += level2_low;
        
        precision highp int;
        int level2_high;
        sum += level2_high;
        do {
            if (true) {
                precision mediump int;
                int level4_medium;
                sum += level4_medium;
            }
            int level3_high;
            sum += level3_high;
        } while (true);	
        int level2_high2;
        sum += level2_high2;
    }
    int level1_low3;
    sum += level1_low3;

    sum += 4 + ((ivec2(level1_low3) * ivec2(level1_high) + ivec2((/* comma operator */level1_low3, level1_high)))).x;
    
    texture2D(samplerLow, vec2(0.1, 0.2));
    texture2D(samplerMed, vec2(0.1, 0.2));
    texture2D(samplerHigh, vec2(0.1, 0.2));
}

precision mediump bool;                 // ERROR
//precision mediump struct { int a; } s;  // ERROR
struct s {int a;};
precision mediump s;                    // ERROR
mediump bvec2 b2;                       // ERROR
