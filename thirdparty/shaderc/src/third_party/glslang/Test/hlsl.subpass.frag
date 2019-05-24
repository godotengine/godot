
layout(input_attachment_index = 1) SubpassInput<float4> subpass_f4 : register(t1);
layout(input_attachment_index = 2) SubpassInput<int4>   subpass_i4;
layout(input_attachment_index = 3) SubpassInput<uint4>  subpass_u4;

layout(input_attachment_index = 4) SubpassInputMS<float4> subpass_ms_f4;
layout(input_attachment_index = 5) SubpassInputMS<int4>   subpass_ms_i4;
layout(input_attachment_index = 6) SubpassInputMS<uint4>  subpass_ms_u4;

layout(input_attachment_index = 1) SubpassInput<float3> subpass_f3;
layout(input_attachment_index = 2) SubpassInput<int3>   subpass_i3;
layout(input_attachment_index = 3) SubpassInput<uint3>  subpass_u3;

layout(input_attachment_index = 4) SubpassInputMS<float3> subpass_ms_f3;
layout(input_attachment_index = 5) SubpassInputMS<int3>   subpass_ms_i3;
layout(input_attachment_index = 6) SubpassInputMS<uint3>  subpass_ms_u3;

layout(input_attachment_index = 1) SubpassInput<float2> subpass_f2;
layout(input_attachment_index = 2) SubpassInput<int2>   subpass_i2;
layout(input_attachment_index = 3) SubpassInput<uint2>  subpass_u2;

layout(input_attachment_index = 4) SubpassInputMS<float2> subpass_ms_f2;
layout(input_attachment_index = 5) SubpassInputMS<int2>   subpass_ms_i2;
layout(input_attachment_index = 6) SubpassInputMS<uint2>  subpass_ms_u2;

layout(input_attachment_index = 1) SubpassInput<float> subpass_f;
layout(input_attachment_index = 2) SubpassInput<int>   subpass_i;
layout(input_attachment_index = 3) SubpassInput<uint>  subpass_u;

layout(input_attachment_index = 4) SubpassInputMS<float> subpass_ms_f;
layout(input_attachment_index = 5) SubpassInputMS<int>   subpass_ms_i;
layout(input_attachment_index = 6) SubpassInputMS<uint>  subpass_ms_u;

[[vk::input_attachment_index(7)]] SubpassInput subpass_2;

struct mystruct_f_t
{
    float  c0;
    float2 c1;
    float  c2;
};

struct mystruct_i_t
{
    int  c0;
    int2 c1;
    int  c2;
};

struct mystruct_u_t
{
    uint  c0;
    uint2 c1;
    uint  c2;
};

// TODO: ...
// layout(input_attachment_index = 7) SubpassInput<mystruct_f_t>    subpass_fs;
// layout(input_attachment_index = 8) SubpassInputMS<mystruct_f_t>  subpass_ms_fs;

// layout(input_attachment_index = 7) SubpassInput<mystruct_i_t>    subpass_is;
// layout(input_attachment_index = 8) SubpassInputMS<mystruct_i_t>  subpass_ms_is;

// layout(input_attachment_index = 7) SubpassInput<mystruct_u_t>    subpass_us;
// layout(input_attachment_index = 8) SubpassInputMS<mystruct_u_t>  subpass_ms_us;

float4 main() : SV_Target0
{
    float4 result00 = subpass_f4.SubpassLoad();
    int4   result01 = subpass_i4.SubpassLoad();
    uint4  result02 = subpass_u4.SubpassLoad();

    float4 result10 = subpass_ms_f4.SubpassLoad(3);
    int4   result11 = subpass_ms_i4.SubpassLoad(3);
    uint4  result12 = subpass_ms_u4.SubpassLoad(3);

    float3 result20 = subpass_f3.SubpassLoad();
    int3   result21 = subpass_i3.SubpassLoad();
    uint3  result22 = subpass_u3.SubpassLoad();

    float3 result30 = subpass_ms_f3.SubpassLoad(3);
    int3   result31 = subpass_ms_i3.SubpassLoad(3);
    uint3  result32 = subpass_ms_u3.SubpassLoad(3);

    float2 result40 = subpass_f2.SubpassLoad();
    int2   result41 = subpass_i2.SubpassLoad();
    uint2  result42 = subpass_u2.SubpassLoad();

    float2 result50 = subpass_ms_f2.SubpassLoad(2);
    int2   result51 = subpass_ms_i2.SubpassLoad(2);
    uint2  result52 = subpass_ms_u2.SubpassLoad(2);

    float  result60 = subpass_f.SubpassLoad();
    int    result61 = subpass_i.SubpassLoad();
    uint   result62 = subpass_u.SubpassLoad();

    float  result70 = subpass_ms_f.SubpassLoad(2);
    int    result71 = subpass_ms_i.SubpassLoad(2);
    uint   result72 = subpass_ms_u.SubpassLoad(2);

    float4 result73 = subpass_2.SubpassLoad();

    // TODO: 
    // mystruct_f_t result80 = subpass_fs.SubpassLoad();
    // mystruct_i_t result81 = subpass_is.SubpassLoad();
    // mystruct_u_t result82 = subpass_us.SubpassLoad();

    // mystruct_f_t result90 = subpass_ms_sf.SubpassLoad(2);
    // mystruct_i_t result91 = subpass_ms_if.SubpassLoad(2);
    // mystruct_u_t result92 = subpass_ms_uf.SubpassLoad(2);

    return 0;
}
