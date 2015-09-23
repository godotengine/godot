#pragma once
#include <map>
#include <string>
#include "stdint_defs.h"

namespace pe_bliss
{
	//Typedef for version info functions: Name - Value
	typedef std::map<std::wstring, std::wstring> string_values_map;
	//Typedef for version info functions: Language string - String Values Map
	//Language String consists of LangID and CharsetID
	//E.g. 041904b0 for Russian UNICODE, 040004b0 for Process Default Language UNICODE
	typedef std::map<std::wstring, string_values_map> lang_string_values_map;

	//Typedef for version info functions: Language - Character Set
	typedef std::multimap<uint16_t, uint16_t> translation_values_map;
}
