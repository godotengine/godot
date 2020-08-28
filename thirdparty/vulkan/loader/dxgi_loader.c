#include "dxgi_loader.h"

#include <strsafe.h>

static HMODULE load_dxgi_module() {
    TCHAR systemPath[MAX_PATH] = "";
    GetSystemDirectory(systemPath, MAX_PATH);
    StringCchCat(systemPath, MAX_PATH, TEXT("\\dxgi.dll"));

    return LoadLibrary(systemPath);
}

typedef HRESULT (APIENTRY *PFN_CreateDXGIFactory1)(REFIID riid, void **ppFactory);

HRESULT dyn_CreateDXGIFactory1(REFIID riid, void **ppFactory) {
    PFN_CreateDXGIFactory1 fpCreateDXGIFactory1 =
        (PFN_CreateDXGIFactory1)GetProcAddress(load_dxgi_module(), "CreateDXGIFactory1");

    if (fpCreateDXGIFactory1 != NULL)
        return fpCreateDXGIFactory1(riid, ppFactory);

    return DXGI_ERROR_NOT_FOUND;
}