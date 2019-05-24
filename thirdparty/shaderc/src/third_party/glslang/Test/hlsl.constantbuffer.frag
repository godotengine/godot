
struct c1_t {
    float4 x;
};

struct c2_t {
    bool x;
    float y;
};

ConstantBuffer<c1_t> cb1 : register(b12);
ConstantBuffer<c2_t> cb2[3];
ConstantBuffer<c2_t> cb3[2][4];

cbuffer cbuff {
    int c1;
};

float4 main() : SV_Target0
{
    if (cb3[1][2].x)
        return cb1.x + cb2[1].y + c1;
    else
        return cb3[1][3].y;
}

