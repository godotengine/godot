#ifndef DXGI_LOADER_H
#define DXGI_LOADER_H

#include <dxgi1_2.h>

HRESULT dyn_CreateDXGIFactory1(REFIID riid, void **ppFactory);

#endif