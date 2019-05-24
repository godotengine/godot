#version 130

uniform sampler1D       texSampler1D;
uniform sampler2D       texSampler2D;
uniform sampler3D       texSampler3D;
uniform samplerCube	    texSamplerCube;
uniform sampler1DShadow shadowSampler1D;
uniform sampler2DShadow shadowSampler2D;

uniform float blend;
uniform vec2 scale;
uniform vec4 u;

varying vec2 t;
varying vec2 coords2D;

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

    color += texture1D    (texSampler1D, coords1D);
    color += texture1D    (texSampler1D, coords1D, bias);
    color += texture1DProj(texSampler1D, coords2D);
    color += texture1DProj(texSampler1D, coords4D);
    color += texture1DProj(texSampler1D, coords2D, bias);
    color += texture1DProj(texSampler1D, coords4D, bias);
    
    color += texture2D        (texSampler2D, coords2D);
    color += texture2D        (texSampler2D, coords2D, bias);
    color += texture2DProj    (texSampler2D, coords3D);
    color += texture2DProj    (texSampler2D, coords4D, bias);

    color += texture3D        (texSampler3D, coords3D);
    color += texture3D        (texSampler3D, coords3D, bias);
    color += texture3DProj    (texSampler3D, coords4D);
    color += texture3DProj    (texSampler3D, coords4D, bias);

    color += textureCube    (texSamplerCube, coords3D);
    color += textureCube    (texSamplerCube, coords3D, bias);
    
    color += shadow1D       (shadowSampler1D, coords3D);
    color += shadow1D       (shadowSampler1D, coords3D, bias);
    color += shadow2D       (shadowSampler2D, coords3D);
    color += shadow2D       (shadowSampler2D, coords3D, bias);
    color += shadow1DProj   (shadowSampler1D, coords4D);
    color += shadow1DProj   (shadowSampler1D, coords4D, bias);
    color += shadow2DProj   (shadowSampler2D, coords4D);
    color += shadow2DProj   (shadowSampler2D, coords4D, bias);

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