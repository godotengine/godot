#version 400

uniform sampler s;
uniform samplerShadow sShadow;
uniform sampler sA[4];
uniform texture2D t2d;
uniform texture3D t3d[4];
flat in int i;

out vec4 color;

void main()
{
    color = texture(sampler2D(t2d, s), vec2(0.5));
    color += texture(sampler3D(t3d[i], sA[2]), vec3(0.5));
    color += texture(sampler2D(t2d, s), vec2(0.5));
}

uniform texture2D                 tex2D;
uniform textureCube               texCube;
uniform textureCubeArray          texCubeArray;
uniform itextureCubeArray         itexCubeArray;
uniform utextureCubeArray         utexCubeArray;
uniform itexture1DArray           itex1DArray;
uniform utexture1D                utex1D;
uniform itexture1D                itex1D;
uniform utexture1DArray           utex1DArray;
uniform textureBuffer             texBuffer;
uniform texture2DArray            tex2DArray;
uniform itexture2D                itex2D;
uniform itexture3D                itex3D;
uniform itextureCube              itexCube;
uniform itexture2DArray           itex2DArray;
uniform utexture2D                utex2D;
uniform utexture3D                utex3D;
uniform utextureCube              utexCube;
uniform utexture2DArray           utex2DArray;
uniform itexture2DRect            itex2DRect;
uniform utexture2DRect            utex2DRect;
uniform itextureBuffer            itexBuffer;
uniform utextureBuffer            utexBuffer;
uniform texture2DMS               tex2DMS;
uniform itexture2DMS              itex2DMS;
uniform utexture2DMS              utex2DMS;
uniform texture2DMSArray          tex2DMSArray;
uniform itexture2DMSArray         itex2DMSArray;
uniform utexture2DMSArray         utex2DMSArray;
uniform texture1D                 tex1D;
uniform texture3D                 tex3D;
uniform texture2DRect             tex2DRect;
uniform texture1DArray            tex1DArray;

void foo()
{
    sampler2D              (tex2D, s);
    samplerCube            (texCube, s);
    samplerCubeArray       (texCubeArray, s);
    samplerCubeArrayShadow (texCubeArray, sShadow);
    isamplerCubeArray      (itexCubeArray, s);
    usamplerCubeArray      (utexCubeArray, s);
    sampler1DArrayShadow   (tex1DArray, sShadow);
    isampler1DArray        (itex1DArray, s);
    usampler1D             (utex1D, s);
    isampler1D             (itex1D, s);
    usampler1DArray        (utex1DArray, s);
    samplerBuffer          (texBuffer, s);
    samplerCubeShadow      (texCube, sShadow);
    sampler2DArray         (tex2DArray, s);
    sampler2DArrayShadow   (tex2DArray, sShadow);
    isampler2D             (itex2D, s);
    isampler3D             (itex3D, s);
    isamplerCube           (itexCube, s);
    isampler2DArray        (itex2DArray, s);
    usampler2D             (utex2D, s);
    usampler3D             (utex3D, s);
    usamplerCube           (utexCube, s);
    usampler2DArray        (utex2DArray, s);
    isampler2DRect         (itex2DRect, s);
    usampler2DRect         (utex2DRect, s);
    isamplerBuffer         (itexBuffer, s);
    usamplerBuffer         (utexBuffer, s);
    sampler2DMS            (tex2DMS, s);
    isampler2DMS           (itex2DMS, s);
    usampler2DMS           (utex2DMS, s);
    sampler2DMSArray       (tex2DMSArray, s);
    isampler2DMSArray      (itex2DMSArray, s);
    usampler2DMSArray      (utex2DMSArray, s);
    sampler1D              (tex1D, s);
    sampler1DShadow        (tex1D, sShadow);
    sampler3D              (tex3D, s);
    sampler2DShadow        (tex2D, sShadow);
    sampler2DRect          (tex2DRect, s);
    sampler2DRectShadow    (tex2DRect, sShadow);
    sampler1DArray         (tex1DArray, s);
}
