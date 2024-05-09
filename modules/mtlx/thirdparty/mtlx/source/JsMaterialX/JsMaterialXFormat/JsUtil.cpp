#include <JsMaterialX/Helpers.h>
#include <MaterialXFormat/Util.h>

#include <emscripten/bind.h>

namespace ems = emscripten;
namespace mx = MaterialX;

EMSCRIPTEN_BINDINGS(xformatUtil)
{
  ems::constant("PATH_LIST_SEPARATOR", mx::PATH_LIST_SEPARATOR);
  ems::constant("MATERIALX_SEARCH_PATH_ENV_VAR", mx::MATERIALX_SEARCH_PATH_ENV_VAR);
}

