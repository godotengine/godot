// Test if including windows.h conflicts with httplib.h

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <httplib.h>
