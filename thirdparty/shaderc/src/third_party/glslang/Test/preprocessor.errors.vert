#version 310 es

#define X 1

#if X
  #ifdef Y
    #error This should not show up in pp output.
  #endif
    #error This should show up in pp output.
#else
  #error This should not show up in pp output.
#endif

#def X
#if Y

#extension a

int main() {
}
