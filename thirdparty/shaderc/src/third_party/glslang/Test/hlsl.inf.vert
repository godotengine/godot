float4 main() : SV_Position
{
    float f1 = -1.#INF;
    float f2 =  1.#INF;
    float f3 = +1.#INF;
    float f4 = f2 * 1.#INF + 1.#INF;
    const float f5 = -1.#INF;
    const float f6 = f5 * 0.0f;

    return (float4)(f1 + f2 + f3 + f4 + f5 + f6);
}