
struct foo_t {
    float float;
};

float fn(float float) { return float; }

float4 main() : SV_Target0
{
    float float = 7;
    bool bool[2] = { float, float };
    int int   = bool[1];
    uint uint = float + int;
    min16float min16float = uint;
    min10float min10float = min16float;
    half half = 0.5;

    {
        foo_t float;
        float.float = 42;
    }

    bool[0] = bool[1];

    float = float + int + uint + min16float + min10float + (bool[0] ? int : float) + fn(float);

    half2x3 half2x3;
    half2x3._11 = (float) * float;

    return float + half2x3._11;
}
