#version 140

uniform sampler1D       texSampler1D;
uniform sampler2D       texSampler2D;
uniform sampler3D       texSampler3D;
uniform samplerCube	    texSamplerCube;
uniform sampler1DShadow shadowSampler1D;
uniform sampler2DShadow shadowSampler2D;

varying float blend;
varying vec2 scale;
varying vec4 u;

in  vec2 t;
in  vec2 coords2D;

void main()
{  
    float blendscale = 1.789;
    float bias       = 2.0;
    float lod		 = 3.0;
    float proj       = 2.0;
    float coords1D   = 1.789;
    vec3  coords3D   = vec3(1.789, 2.718, 3.453);
    vec4  coords4D   = vec4(1.789, 2.718, 3.453, 2.0);
    vec4  color      = vec4(0.0, 0.0, 0.0, 0.0);

    color += texture    (texSampler1D, coords1D);
    color += texture    (texSampler1D, coords1D, bias);
    color += textureProj(texSampler1D, coords2D);
    color += textureProj(texSampler1D, coords4D);
    color += textureProj(texSampler1D, coords2D, bias);
    color += textureProj(texSampler1D, coords4D, bias);
    
    color += texture        (texSampler2D, coords2D);
    color += texture        (texSampler2D, coords2D, bias);
    color += textureProj    (texSampler2D, coords3D);
    color += textureProj    (texSampler2D, coords4D, bias);

    color += texture        (texSampler3D, coords3D);
    color += texture        (texSampler3D, coords3D, bias);
    color += textureProj    (texSampler3D, coords4D);
    color += textureProj    (texSampler3D, coords4D, bias);

    color += texture    (texSamplerCube, coords3D);
    color += texture    (texSamplerCube, coords3D, bias);
    
    color += texture       (shadowSampler1D, coords3D);
    color += texture       (shadowSampler1D, coords3D, bias);
    color += texture       (shadowSampler2D, coords3D);
    color += texture       (shadowSampler2D, coords3D, bias);
    color += textureProj   (shadowSampler1D, coords4D);
    color += textureProj   (shadowSampler1D, coords4D, bias);
    color += textureProj   (shadowSampler2D, coords4D);
    color += textureProj   (shadowSampler2D, coords4D, bias);

    ivec2 iCoords2D = ivec2(0, 5);
    int iLod = 1;

    color += texelFetch(texSampler2D, iCoords2D, iLod);

    vec2 gradX = dFdx(coords2D);
    vec2 gradY = dFdy(coords2D);
    const ivec2 offset = ivec2(3, -7);

    color += textureGrad(texSampler2D, coords2D, gradX, gradY);
    color += textureProjGrad(texSampler2D, vec3(coords2D, proj), gradX, gradY);
    color += textureGradOffset(texSampler2D, coords2D, gradX, gradY, offset);
    color += textureProjGradOffset(texSampler2D, coords3D, gradX, gradY, offset);
    color += textureGrad(shadowSampler2D, vec3(coords2D, lod), gradX, gradY);
    
    gl_FragColor = mix(color, u, blend * blendscale);
}