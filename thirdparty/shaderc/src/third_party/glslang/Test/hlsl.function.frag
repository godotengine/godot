float4 fun0()
{
    return 1.0f;
}

uint fun2(float4 col)
{
    return 7;
}

float4 fun4(uint id1, uniform uint id2)
{
    return id1 * id2;
}

float4 fun1(int index)
{
    uint entityId = fun2(fun0());
    return fun4(entityId, entityId);
}

int main() : SV_TARGET
{
    return fun1;
}