#version 310 es
precision mediump float;
in lowp float lowfin;
in mediump float mediumfin;
in highp vec4 highfin;

highp int uniform_high;
mediump int uniform_medium;
lowp int uniform_low;
bvec2 ub2;

out mediump vec4 mediumfout;

highp float global_highp;

lowp vec2 foo(mediump vec3 mv3)
{
    return highfin.xy;
}

bool boolfun(bvec2 bv2)
{
    return bv2 == bvec2(false, true);
}

struct S {
    highp float a;
    lowp float b;
};

in S s;

void main()
{
    lowp int sum = uniform_medium + uniform_high;

    sum += uniform_high;
    sum += uniform_low;
    
    // test maxing precisions of args to get precision of builtin
    lowp float arg1 = 3.2;
    mediump float arg2 = 1023908.2;
    lowp float d = distance(lowfin, mediumfin);

    global_highp = length(highfin);

    highp vec4 local_highp = vec4(global_highp);

    mediumfout = vec4(sin(d)) + arg2 + local_highp;

    sum += 4 + ((ivec2(uniform_low) * ivec2(uniform_high) + ivec2((/* comma operator */uniform_low, uniform_high)))).x;

    mediumfout += vec4(sum);

    if (boolfun(ub2))
        ++mediumfout;
    
    mediumfout *= s.a;
    mediumfout *= s.b;
}
