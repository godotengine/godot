#pragma pack_matrix( row_major )

cbuffer VertexShaderConstants : register(b0)
{
    matrix model;
    matrix projectionAndView;
};

#define ColorRS \
    "RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
    "DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
    "DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
    "DENY_HULL_SHADER_ROOT_ACCESS )," \
    "RootConstants(num32BitConstants=32, b0)," \
    "RootConstants(num32BitConstants=28, b1)"\

#define TextureRS \
    "RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
    "            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
    "            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
    "            DENY_HULL_SHADER_ROOT_ACCESS )," \
    "RootConstants(num32BitConstants=32, b0),"\
    "RootConstants(num32BitConstants=28, b1),"\
    "DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
    "DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )"

#define AdvancedRS \
    "RootFlags ( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |" \
    "            DENY_DOMAIN_SHADER_ROOT_ACCESS |" \
    "            DENY_GEOMETRY_SHADER_ROOT_ACCESS |" \
    "            DENY_HULL_SHADER_ROOT_ACCESS )," \
    "RootConstants(num32BitConstants=32, b0),"\
    "RootConstants(num32BitConstants=28, b1),"\
    "DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
    "DescriptorTable ( SRV(t1), visibility = SHADER_VISIBILITY_PIXEL ),"\
    "DescriptorTable ( SRV(t2), visibility = SHADER_VISIBILITY_PIXEL ),"\
    "DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )"
