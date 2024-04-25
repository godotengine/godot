struct MetalTexture
{
    texture2d<float> tex;
    sampler s;
    int get_width() { return tex.get_width(); }
    int get_height() { return tex.get_height(); }
    int get_num_mip_levels() { return tex.get_num_mip_levels(); }
};

int get_width(MetalTexture mtlTex) { return mtlTex.get_width(); }

float4 texture(MetalTexture mtlTex, float2 uv)
{
    return mtlTex.tex.sample(mtlTex.s, uv);
}

float4 textureLod(MetalTexture mtlTex, float2 uv, float lod)
{
    return mtlTex.tex.sample(mtlTex.s, uv, level(lod));
}

int2 textureSize(MetalTexture mtlTex, int mipLevel)
{
    return int2(mtlTex.get_width(), mtlTex.get_height());
}

int texture_mips(MetalTexture mtlTex)
{
    return mtlTex.tex.get_num_mip_levels();
}
