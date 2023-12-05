proto = """
#define GDVIRTUAL$VER($RET m_name $ARG) \\
StringName _gdvirtual_##m_name##_sn = #m_name;\\
mutable bool _gdvirtual_##m_name##_initialized = false;\\
mutable void* _gdvirtual_##m_name = nullptr;\\
template<bool required>\\
_FORCE_INLINE_ bool _gdvirtual_##m_name##_call($CALLARGS) $CONST { \\
	ScriptInstance *_script_instance = ((Object*)(this))->get_script_instance();\\
	if (_script_instance) {\\
		Callable::CallError ce; \\
		$CALLSIARGS\\
		$CALLSIBEGIN_script_instance->callp(_gdvirtual_##m_name##_sn, $CALLSIARGPASS, ce);\\
		if (ce.error == Callable::CallError::CALL_OK) {\\
			$CALLSIRET\\
			return true;\\
		}    \\
	}\\
    if (unlikely(_get_extension() && !_gdvirtual_##m_name##_initialized)) {\\
        _gdvirtual_##m_name = nullptr;\\
        if (_get_extension()->get_virtual_call_data && _get_extension()->call_virtual_with_data) {\\
            _gdvirtual_##m_name = _get_extension()->get_virtual_call_data(_get_extension()->class_userdata, &_gdvirtual_##m_name##_sn);\\
        } else if (_get_extension()->get_virtual) {\\
            _gdvirtual_##m_name = (void *)_get_extension()->get_virtual(_get_extension()->class_userdata, &_gdvirtual_##m_name##_sn);\\
        }\\
        GDVIRTUAL_TRACK(_gdvirtual_##m_name, _gdvirtual_##m_name##_initialized); \\
        _gdvirtual_##m_name##_initialized = true;\\
    }\\
	if (_gdvirtual_##m_name) {\\
		$CALLPTRARGS\\
		$CALLPTRRETDEF\\
        if (_get_extension()->get_virtual_call_data && _get_extension()->call_virtual_with_data) {\\
            _get_extension()->call_virtual_with_data(_get_extension_instance(), &_gdvirtual_##m_name##_sn, _gdvirtual_##m_name, $CALLPTRARGPASS,$CALLPTRRETPASS);\\
            $CALLPTRRET\\
        } else {\\
		    ((GDExtensionClassCallVirtual)_gdvirtual_##m_name)(_get_extension_instance(),$CALLPTRARGPASS,$CALLPTRRETPASS);\\
            $CALLPTRRET\\
        }\\
		return true;\\
	}\\
	\\
	if (required) {\\
	        ERR_PRINT_ONCE("Required virtual method " + get_class() + "::" + #m_name + " must be overridden before calling.");\\
	        $RVOID\\
    }\\
\\
    return false;\\
}\\
_FORCE_INLINE_ bool _gdvirtual_##m_name##_overridden() const { \\
	ScriptInstance *_script_instance = ((Object*)(this))->get_script_instance();\\
	if (_script_instance && _script_instance->has_method(_gdvirtual_##m_name##_sn)) {\\
		return true;\\
	}\\
    if (unlikely(_get_extension() && !_gdvirtual_##m_name##_initialized)) {\\
         _gdvirtual_##m_name = nullptr;\\
        if (_get_extension()->get_virtual_call_data && _get_extension()->call_virtual_with_data) {\\
             _gdvirtual_##m_name = _get_extension()->get_virtual_call_data(_get_extension()->class_userdata, &_gdvirtual_##m_name##_sn);\\
        } else if (_get_extension()->get_virtual) {\\
             _gdvirtual_##m_name = (void *)_get_extension()->get_virtual(_get_extension()->class_userdata, &_gdvirtual_##m_name##_sn);\\
        }\\
        GDVIRTUAL_TRACK(_gdvirtual_##m_name, _gdvirtual_##m_name##_initialized); \\
        _gdvirtual_##m_name##_initialized = true;\\
    }\\
	if (_gdvirtual_##m_name) {\\
	    return true;\\
	}\\
	return false;\\
}\\
\\
_FORCE_INLINE_ static MethodInfo _gdvirtual_##m_name##_get_method_info() { \\
    MethodInfo method_info;\\
    method_info.name = #m_name;\\
    method_info.flags = METHOD_FLAG_VIRTUAL;\\
    $FILL_METHOD_INFO\\
    return method_info;\\
}

"""


def generate_version(argcount, const=False, returns=False):
    s = proto
    sproto = str(argcount)
    method_info = ""
    if returns:
        sproto += "R"
        s = s.replace("$RET", "m_ret, ")
        s = s.replace("$RVOID", "(void)r_ret;")  # If required, may lead to uninitialized errors
        s = s.replace("$CALLPTRRETDEF", "PtrToArg<m_ret>::EncodeT ret;")
        method_info += "\tmethod_info.return_val = GetTypeInfo<m_ret>::get_class_info();\\\n"
        method_info += "\tmethod_info.return_val_metadata = GetTypeInfo<m_ret>::METADATA;\\\n"
    else:
        s = s.replace("$RET", "")
        s = s.replace("$RVOID", "")
        s = s.replace("$CALLPTRRETDEF", "")

    if const:
        sproto += "C"
        s = s.replace("$CONST", "const")
        method_info += "\tmethod_info.flags|=METHOD_FLAG_CONST;\\\n"
    else:
        s = s.replace("$CONST", "")

    s = s.replace("$VER", sproto)
    argtext = ""
    callargtext = ""
    callsiargs = ""
    callsiargptrs = ""
    callptrargsptr = ""
    if argcount > 0:
        argtext += ", "
        callsiargs = "Variant vargs[" + str(argcount) + "]={"
        callsiargptrs = "\t\tconst Variant *vargptrs[" + str(argcount) + "]={"
        callptrargsptr = "\t\tGDExtensionConstTypePtr argptrs[" + str(argcount) + "]={"
    callptrargs = ""
    for i in range(argcount):
        if i > 0:
            argtext += ", "
            callargtext += ", "
            callsiargs += ", "
            callsiargptrs += ", "
            callptrargs += "\t\t"
            callptrargsptr += ", "
        argtext += "m_type" + str(i + 1)
        callargtext += "m_type" + str(i + 1) + " arg" + str(i + 1)
        callsiargs += "Variant(arg" + str(i + 1) + ")"
        callsiargptrs += "&vargs[" + str(i) + "]"
        callptrargs += (
            "PtrToArg<m_type" + str(i + 1) + ">::EncodeT argval" + str(i + 1) + " = arg" + str(i + 1) + ";\\\n"
        )
        callptrargsptr += "&argval" + str(i + 1)
        method_info += "\tmethod_info.arguments.push_back(GetTypeInfo<m_type" + str(i + 1) + ">::get_class_info());\\\n"
        method_info += (
            "\tmethod_info.arguments_metadata.push_back(GetTypeInfo<m_type" + str(i + 1) + ">::METADATA);\\\n"
        )

    if argcount:
        callsiargs += "};\\\n"
        callsiargptrs += "};\\\n"
        s = s.replace("$CALLSIARGS", callsiargs + callsiargptrs)
        s = s.replace("$CALLSIARGPASS", "(const Variant **)vargptrs," + str(argcount))
        callptrargsptr += "};\\\n"
        s = s.replace("$CALLPTRARGS", callptrargs + callptrargsptr)
        s = s.replace("$CALLPTRARGPASS", "reinterpret_cast<GDExtensionConstTypePtr*>(argptrs)")
    else:
        s = s.replace("$CALLSIARGS", "")
        s = s.replace("$CALLSIARGPASS", "nullptr, 0")
        s = s.replace("$CALLPTRARGS", "")
        s = s.replace("$CALLPTRARGPASS", "nullptr")

    if returns:
        if argcount > 0:
            callargtext += ","
        callargtext += " m_ret& r_ret"
        s = s.replace("$CALLSIBEGIN", "Variant ret = ")
        s = s.replace("$CALLSIRET", "r_ret = VariantCaster<m_ret>::cast(ret);")
        s = s.replace("$CALLPTRRETPASS", "&ret")
        s = s.replace("$CALLPTRRET", "r_ret = (m_ret)ret;")
    else:
        s = s.replace("$CALLSIBEGIN", "")
        s = s.replace("$CALLSIRET", "")
        s = s.replace("$CALLPTRRETPASS", "nullptr")
        s = s.replace("$CALLPTRRET", "")

    s = s.replace("$ARG", argtext)
    s = s.replace("$CALLARGS", callargtext)
    s = s.replace("$FILL_METHOD_INFO", method_info)

    return s


def run(target, source, env):
    max_versions = 12

    txt = """
#ifndef GDVIRTUAL_GEN_H
#define GDVIRTUAL_GEN_H

#include "core/object/script_instance.h"

#ifdef TOOLS_ENABLED
#define GDVIRTUAL_TRACK(m_virtual, m_initialized) \\
    if (_get_extension()->reloadable) {\\
        VirtualMethodTracker *tracker = memnew(VirtualMethodTracker);\\
        tracker->method = (void **)&m_virtual;\\
        tracker->initialized = &m_initialized;\\
        tracker->next = virtual_method_list;\\
        virtual_method_list = tracker;\\
    }
#else
#define GDVIRTUAL_TRACK(m_virtual, m_initialized)
#endif

"""

    for i in range(max_versions + 1):
        txt += "/* " + str(i) + " Arguments */\n\n"
        txt += generate_version(i, False, False)
        txt += generate_version(i, False, True)
        txt += generate_version(i, True, False)
        txt += generate_version(i, True, True)

    txt += "#endif"

    with open(target[0], "w") as f:
        f.write(txt)


if __name__ == "__main__":
    from platform_methods import subprocess_main

    subprocess_main(globals())
