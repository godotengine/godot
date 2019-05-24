#version 140

in float blend;
in vec4 u;
bool p;

in vec2 t;

void main()
{
    float blendscale = 1.789;

    vec4 w = u;
    vec4 w_undef;       // test undef
    vec4 w_dep = u;     // test dependent swizzles
    vec4 w_reorder = u; // test reordering
    vec4 w2 = u;
    vec4 w_flow = u;    // test flowControl

    w_reorder.z = blendscale;

    w.wy = t;

    w_reorder.x = blendscale;

    w2.xyzw = u.zwxy;

    w_reorder.y = blendscale;

    w_dep.xy = w2.xz;
    w_dep.zw = t;

    w_undef.xy = u.zw;

    if (p)
        w_flow.x = t.x;
    else
        w_flow.x = t.y;

    gl_FragColor = mix(w_reorder, w_undef, w * w2 * w_dep * w_flow);

    vec2 c = t;
    vec4 rep = vec4(0.0, 0.0, 0.0, 1.0);

    if (c.x < 0.0)
        c.x *= -1.0;

    if (c.x <= 1.0)
        rep.x = 3.4;

    gl_FragColor += rep;
}
