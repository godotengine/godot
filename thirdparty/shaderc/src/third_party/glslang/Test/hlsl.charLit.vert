float4 main() : SV_Position
{
    uint a1 =  'A';
    int  a2 =  '0';

    int  a3 = '\a';
    a3     += '\b';
    a3     += '\t';
    a3     += '\n';
    a3     += '\v';
    a3     += '\f';
    a3     += '\r';

    int a10 = '\c';

    return a1 + a2 + a3 + a10;
}
