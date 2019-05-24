
// Test binding autoassignment and offset for SubpassInput objects

layout(input_attachment_index = 1) SubpassInput<float4> subpass_f4 : register(t1);
layout(input_attachment_index = 4) SubpassInputMS<float4> subpass_ms_f4;
[[vk::input_attachment_index(7)]] SubpassInput subpass_2;

float4 main() : SV_Target0
{
    float4 result00 = subpass_f4.SubpassLoad();
    float4 result10 = subpass_ms_f4.SubpassLoad(3);
    float4 result73 = subpass_2.SubpassLoad();

    return 0;
}
