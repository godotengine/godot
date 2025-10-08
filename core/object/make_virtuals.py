script_call = """ScriptInstance *_script_instance = ((Object *)(this))->get_script_instance();\\
		if (_script_instance) {\\
			Callable::CallError ce;\\
			$CALLSIARGS\\
			$CALLSIBEGIN_script_instance->callp(_gdvirtual_##$VARNAME##_sn, $CALLSIARGPASS, ce);\\
			if (ce.error == Callable::CallError::CALL_OK) {\\
				$CALLSIRET\\
				return true;\\
			}\\
		}"""

script_has_method = """ScriptInstance *_script_instance = ((Object *)(this))->get_script_instance();\\
		if (_script_instance && _script_instance->has_method(_gdvirtual_##$VARNAME##_sn)) {\\
			return true;\\
		}"""

proto = """\
#define _GDVIRTUAL$VER_CALL($ALIAS $RET m_name $ARG)\\
	static const StringName _gdvirtual_##$VARNAME##_sn = StringName(#m_name, true);\\
	$SCRIPTCALL\\
	if (_get_extension()) {\\
		if (unlikely(!_gdvirtual_##$VARNAME)) {\\
			_gdvirtual_init_method_ptr(_gdvirtual_##$VARNAME##_get_method_info().get_compatibility_hash(), _gdvirtual_##$VARNAME, _gdvirtual_##$VARNAME##_sn, $COMPAT);\\
		}\\
		if (_gdvirtual_##$VARNAME != reinterpret_cast<void*>(_INVALID_GDVIRTUAL_FUNC_ADDR)) {\\
			$CALLPTRARGS\\
			$CALLPTRRETDEF\\
			if (_get_extension()->call_virtual_with_data) {\\
				_get_extension()->call_virtual_with_data(_get_extension_instance(), &_gdvirtual_##$VARNAME##_sn, _gdvirtual_##$VARNAME, $CALLPTRARGPASS, $CALLPTRRETPASS);\\
				$CALLPTRRET\\
			} else {\\
				((GDExtensionClassCallVirtual)_gdvirtual_##$VARNAME)(_get_extension_instance(), $CALLPTRARGPASS, $CALLPTRRETPASS);\\
				$CALLPTRRET\\
			}\\
			return true;\\
		}\\
	}\\
	$REQCHECK\\
	$RVOID\\
	return false;
#define _GDVIRTUAL$VER_OVERRIDE($ALIAS $RET m_name $ARG)\\
	static const StringName _gdvirtual_##$VARNAME##_sn = StringName(#m_name, true);\\
	$SCRIPTHASMETHOD\\
	if (_get_extension()) {\\
		if (unlikely(!_gdvirtual_##$VARNAME)) {\\
			_gdvirtual_init_method_ptr(_gdvirtual_##$VARNAME##_get_method_info().get_compatibility_hash(), _gdvirtual_##$VARNAME, _gdvirtual_##$VARNAME##_sn, $COMPAT);\\
		}\\
		if (_gdvirtual_##$VARNAME != reinterpret_cast<void*>(_INVALID_GDVIRTUAL_FUNC_ADDR)) {\\
			return true;\\
		}\\
	}\\
	return false;
#define _GDVIRTUAL$VER_GET_METHOD_INFO($ALIAS $RET m_name $ARG)\\
	MethodInfo method_info;\\
	method_info.name = #m_name;\\
	method_info.flags = $METHOD_FLAGS;\\
	$FILL_METHOD_INFO\\
	return method_info;
#define GDVIRTUAL$VER($ALIAS $RET m_name $ARG)\\
	mutable void *_gdvirtual_##$VARNAME = nullptr;\\
	_FORCE_INLINE_ bool _gdvirtual_##$VARNAME##_call($CALLARGS) $CONST { _GDVIRTUAL$VER_CALL($ALIAS $RET m_name $ARG) }\\
	_FORCE_INLINE_ bool _gdvirtual_##$VARNAME##_overridden() const { _GDVIRTUAL$VER_OVERRIDE($ALIAS $RET m_name $ARG) }\\
	_FORCE_INLINE_ static MethodInfo _gdvirtual_##$VARNAME##_get_method_info() { _GDVIRTUAL$VER_GET_METHOD_INFO($ALIAS $RET m_name $ARG) }
#define DECLARE_GDVIRTUAL$VER($ALIAS $RET m_name $ARG)\\
	mutable void *_gdvirtual_##$VARNAME = nullptr;\\
	bool _gdvirtual_##$VARNAME##_call($CALLARGS) $CONST ;\\
	bool _gdvirtual_##$VARNAME##_overridden() const;\\
	static MethodInfo _gdvirtual_##$VARNAME##_get_method_info();
#define IMPLEMENT_GDVIRTUAL$VER(m_class, $ALIAS $RET m_name $ARG)\\
	bool m_class::_gdvirtual_##$VARNAME##_call($CALLARGS) $CONST { _GDVIRTUAL$VER_CALL($ALIAS $RET m_name $ARG) }\\
	bool m_class::_gdvirtual_##$VARNAME##_overridden() const { _GDVIRTUAL$VER_OVERRIDE($ALIAS $RET m_name $ARG) }\\
	MethodInfo m_class::_gdvirtual_##$VARNAME##_get_method_info() { _GDVIRTUAL$VER_GET_METHOD_INFO($ALIAS $RET m_name $ARG) }

"""


def generate_version(argcount, const=False, returns=False, required=False, compat=False):
    s = proto
    if compat:
        s = s.replace("\t$SCRIPTCALL\\\n", "")
        s = s.replace("\t$SCRIPTHASMETHOD\\\n", "")
    else:
        s = s.replace("$SCRIPTCALL", script_call)
        s = s.replace("$SCRIPTHASMETHOD", script_has_method)

    sproto = str(argcount)
    method_info = ""
    method_flags = "METHOD_FLAG_VIRTUAL"
    if returns:
        sproto += "R"
        s = s.replace("$RET", "m_ret,")
        s = s.replace("$RVOID", "(void)r_ret;")  # If required, may lead to uninitialized errors
        s = s.replace("$CALLPTRRETDEF", "PtrToArg<m_ret>::EncodeT ret;")
        method_info += "method_info.return_val = GetTypeInfo<m_ret>::get_class_info();\\\n"
        method_info += "\tmethod_info.return_val_metadata = GetTypeInfo<m_ret>::METADATA;"
    else:
        s = s.replace("$RET ", "")
        s = s.replace("\t$RVOID\\\n", "")
        s = s.replace("\t\t$CALLPTRRETDEF\\\n", "")

    if const:
        sproto += "C"
        method_flags += " | METHOD_FLAG_CONST"
        s = s.replace("$CONST", "const")
    else:
        s = s.replace("$CONST ", "")

    if required:
        sproto += "_REQUIRED"
        method_flags += " | METHOD_FLAG_VIRTUAL_REQUIRED"
        s = s.replace(
            "$REQCHECK",
            'ERR_PRINT_ONCE("Required virtual method " + get_class() + "::" + #m_name + " must be overridden before calling.");',
        )
    else:
        s = s.replace("\t$REQCHECK\\\n", "")

    if compat:
        sproto += "_COMPAT"
        s = s.replace("$COMPAT", "true")
        s = s.replace("$ALIAS", "m_alias,")
        s = s.replace("$VARNAME", "m_alias")
    else:
        s = s.replace("$COMPAT", "false")
        s = s.replace("$ALIAS ", "")
        s = s.replace("$VARNAME", "m_name")

    s = s.replace("$METHOD_FLAGS", method_flags)
    s = s.replace("$VER", sproto)
    argtext = ""
    callargtext = ""
    callsiargs = ""
    callsiargptrs = ""
    callptrargsptr = ""
    if argcount > 0:
        argtext += ", "
        callsiargs = f"Variant vargs[{argcount}] = {{ "
        callsiargptrs = f"\t\tconst Variant *vargptrs[{argcount}] = {{ "
        callptrargsptr = f"\t\tGDExtensionConstTypePtr argptrs[{argcount}] = {{ "

        if method_info:
            method_info += "\\\n\t"
        method_info += (
            "_gdvirtual_set_method_info_args<"
            + ", ".join(f"m_type{i + 1}" for i in range(argcount))
            + ">(method_info);"
        )

    callptrargs = ""
    for i in range(argcount):
        if i > 0:
            argtext += ", "
            callargtext += ", "
            callsiargs += ", "
            callsiargptrs += ", "
            callptrargs += "\t\t\t"
            callptrargsptr += ", "
        argtext += f"m_type{i + 1}"
        callargtext += f"m_type{i + 1} arg{i + 1}"
        callsiargs += f"arg{i + 1}"
        callsiargptrs += f"&vargs[{i}]"
        callptrargs += (
            f"PtrToArg<m_type{i + 1}>::EncodeT argval{i + 1} = (PtrToArg<m_type{i + 1}>::EncodeT)arg{i + 1};\\\n"
        )
        callptrargsptr += f"&argval{i + 1}"

    if argcount:
        callsiargs += " };\\\n"
        callsiargptrs += " };"
        s = s.replace("$CALLSIARGS", callsiargs + callsiargptrs)
        s = s.replace("$CALLSIARGPASS", f"(const Variant **)vargptrs, {argcount}")
        callptrargsptr += " };"
        s = s.replace("$CALLPTRARGS", callptrargs + callptrargsptr)
        s = s.replace("$CALLPTRARGPASS", "reinterpret_cast<GDExtensionConstTypePtr *>(argptrs)")
    else:
        s = s.replace("\t\t$CALLSIARGS\\\n", "")
        s = s.replace("$CALLSIARGPASS", "nullptr, 0")
        s = s.replace("\t\t$CALLPTRARGS\\\n", "")
        s = s.replace("$CALLPTRARGPASS", "nullptr")

    if returns:
        if argcount > 0:
            callargtext += ", "
        callargtext += "m_ret &r_ret"
        s = s.replace("$CALLSIBEGIN", "Variant ret = ")
        s = s.replace("$CALLSIRET", "r_ret = VariantCaster<m_ret>::cast(ret);")
        s = s.replace("$CALLPTRRETPASS", "&ret")
        s = s.replace("$CALLPTRRET", "r_ret = (m_ret)ret;")
    else:
        s = s.replace("$CALLSIBEGIN", "")
        s = s.replace("\t\t\t$CALLSIRET\\\n", "")
        s = s.replace("$CALLPTRRETPASS", "nullptr")
        s = s.replace("\t\t\t$CALLPTRRET\\\n", "")

    s = s.replace(" $ARG", argtext)
    s = s.replace("$CALLARGS", callargtext)
    if method_info:
        s = s.replace("$FILL_METHOD_INFO", method_info)
    else:
        s = s.replace("\t$FILL_METHOD_INFO\\\n", method_info)

    return s


def run(target, source, env):
    max_versions = 12

    txt = """/* THIS FILE IS GENERATED DO NOT EDIT */
#pragma once

#include "core/object/script_instance.h"

inline constexpr uintptr_t _INVALID_GDVIRTUAL_FUNC_ADDR = static_cast<uintptr_t>(-1);

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
