#define EMP1(a)
#define EMP2(a, b)

#define EXP1(a) = a
#define EXP2(a, b) = a, b

struct A
{
    float4 a EMP1({1,2,3,4});                           // No PP arg errors
    float4 b EMP2({({{(({1,2,3,4}))}})}, {{1,2,3,4}});  // No PP arg errors
    float4 c EXP1({1,2,3,4});                           // ERROR: No PP arg errors, but init error
    float4 d EXP2({({{(({1,2,3,4}))}})}, {{1,2,3,4}});  // ERROR: No PP arg errors, but init error
};

void main()
{
    "a string"
}
