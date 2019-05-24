#define A defined(B)

#if A
int badGlobal1;
#else
int goodGlobal1;
#endif

#define B

#if A
int goodGlobal2;
#else
int badGlobal2;
#endif

void main() {}
