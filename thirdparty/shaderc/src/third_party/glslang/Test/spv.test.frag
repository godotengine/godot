#version 400

uniform sampler2D texSampler2D;
uniform sampler3D texSampler3D;

in float blend;
in vec2 scale;
in vec4 u;

in vec2 t;
in vec3 coords;

void main()
{  
    float blendscale = 1.789;

    vec4 v = texture(texSampler2D, (t + scale) / scale ).wzyx;

	vec4 w = texture(texSampler3D, coords) + v;
    
    gl_FragColor = mix(w, u, blend * blendscale);
}
