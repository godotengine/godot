proto = """#define GDVIRTUAL$VER($ALIAS $RET m_name $ARG)\\
	class _GDVIRTUAL##$VARNAME: public GDVirtualMethodInfo<$TEMPLATE_ARGS> {\\
	public:\\
		MethodInfo get_method_info() const override {\\
			MethodInfo method_info;\\
			method_info.name = #m_name;\\
			method_info.flags = $METHOD_FLAGS;\\
			$FILL_METHOD_INFO\\
			return method_info;\\
		}\\
		using GDVirtualMethodInfo::GDVirtualMethodInfo;\\
	};\\
	static const _GDVIRTUAL##$VARNAME &get_gdvirtual_##$VARNAME() {\\
		static _GDVIRTUAL##$VARNAME instance($METHOD_INFO_INIT_ARGS);\\
		return instance;\\
	}\\
	mutable void *_gdvirtual_##$VARNAME##_ptr = nullptr;\\

"""


def generate_version(argcount, const=False, returns=False, required=False, compat=False):
    s = proto

    def to_c_bool(b: bool) -> str:
        return "true" if b else "false"

    s = s.replace("$METHOD_INFO_INIT_ARGS", f"#m_name, {to_c_bool(required)}, {to_c_bool(compat)}")

    sproto = str(argcount)
    method_info = ""
    method_flags = "METHOD_FLAG_VIRTUAL"
    template_args = f"{to_c_bool(const)}, "

    if returns:
        sproto += "R"
        s = s.replace("$RET", "m_ret,")
        method_info += "method_info.return_val = GetTypeInfo<m_ret>::get_class_info();\\\n"
        method_info += "\t\t\tmethod_info.return_val_metadata = GetTypeInfo<m_ret>::METADATA;"
        template_args += "m_ret"
    else:
        s = s.replace("$RET ", "")
        template_args += "void"

    if const:
        sproto += "C"
        method_flags += " | METHOD_FLAG_CONST"

    if required:
        sproto += "_REQUIRED"
        method_flags += " | METHOD_FLAG_VIRTUAL_REQUIRED"

    if compat:
        sproto += "_COMPAT"
        s = s.replace("$ALIAS", "m_alias,")
        s = s.replace("$VARNAME", "m_alias")
    else:
        s = s.replace("$ALIAS ", "")
        s = s.replace("$VARNAME", "m_name")

    s = s.replace("$METHOD_FLAGS", method_flags)
    s = s.replace("$VER", sproto)
    argtext = ""
    if argcount > 0:
        argtext += ", "

        if method_info:
            method_info += "\\\n\t\t\t"
        method_info += (
            "_gdvirtual_set_method_info_args<"
            + ", ".join(f"m_type{i + 1}" for i in range(argcount))
            + ">(method_info);"
        )

    for i in range(argcount):
        if i > 0:
            argtext += ", "
        argtext += f"m_type{i + 1}"

    s = s.replace(" $ARG", argtext)
    if method_info:
        s = s.replace("$FILL_METHOD_INFO", method_info)
    else:
        s = s.replace("\t\t\t$FILL_METHOD_INFO\\\n", method_info)

    s = s.replace("$TEMPLATE_ARGS", template_args + argtext)

    return s


def run(target, source, env):
    max_versions = 12

    txt = """/* THIS FILE IS GENERATED DO NOT EDIT */
#pragma once

#include "core/object/script_instance.h"

template <typename... Args>
void _gdvirtual_set_method_info_args(MethodInfo &p_method_info) {
	p_method_info.arguments = { GetTypeInfo<Args>::get_class_info()... };
	p_method_info.arguments_metadata = { GetTypeInfo<Args>::METADATA... };
}

"""

    for i in range(max_versions + 1):
        txt += f"/* {i} Arguments */\n\n"
        txt += generate_version(i, False, False)
        txt += generate_version(i, False, True)
        txt += generate_version(i, True, False)
        txt += generate_version(i, True, True)
        txt += generate_version(i, False, False, True)
        txt += generate_version(i, False, True, True)
        txt += generate_version(i, True, False, True)
        txt += generate_version(i, True, True, True)
        txt += generate_version(i, False, False, False, True)
        txt += generate_version(i, False, True, False, True)
        txt += generate_version(i, True, False, False, True)
        txt += generate_version(i, True, True, False, True)

    with open(str(target[0]), "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
