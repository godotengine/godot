#version 110

uniform sampler2D texSampler2D;
uniform sampler3D texSampler3D;

uniform float blend;
uniform vec2 scale;
uniform vec4 u;

varying vec2 t;
varying vec3 coords;

void main()
{  
    float blendscale = 1.789;

    vec4 v = texture2D(texSampler2D, (t + scale) / scale ).wzyx;

	vec4 w = texture3D(texSampler3D, coords) + v;
    
    gl_FragColor = mix(w, u, blend * blendscale);
}
