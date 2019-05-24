float condf;
int condi;
float1 condf1;
int1 condi1;

float4 main() : SV_Target0
{
    float4 a = float4(2.0, 2.0, 2.0, 2.0);
    if (condi)
        return a + 1.0;
    if (condf)
        return a + 2.0;
    if (condf1)
        return a + 3.0;
    if (condi1)
        return a + 4.0;
    if (condi && condf || condf1)
        return a + 5.0;

    float f = condf;
    while (f) { --f; }

    int i = condi;
    do { --i; } while (i);

    for (; i; ) { --i; }

    float g = condf ? 7.0 : 8.0;
    a += g;

    return a - 1.0;
}