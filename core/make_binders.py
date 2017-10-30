# -*- coding: ibm850 -*-


template_typed = """
#ifdef TYPED_METHOD_BIND
template<class T $ifret ,class R$ $ifargs ,$ $arg, class P@$>
class MethodBind$argc$$ifret R$$ifconst C$ : public MethodBind {
public:

	$ifret R$ $ifnoret void$ (T::*method)($arg, P@$) $ifconst const$;
#ifdef DEBUG_METHODS_ENABLED
	virtual Variant::Type _gen_argument_type(int p_arg) const { return _get_argument_type(p_arg); }
	Variant::Type _get_argument_type(int p_argument) const {
		$ifret if (p_argument==-1) return (Variant::Type)GetTypeInfo<R>::VARIANT_TYPE;$
		$arg if (p_argument==(@-1)) return (Variant::Type)GetTypeInfo<P@>::VARIANT_TYPE;
		$
		return Variant::NIL;
	}
	virtual PropertyInfo _gen_argument_type_info(int p_argument) const {
		$ifret if (p_argument==-1) return GetTypeInfo<R>::get_class_info();$
		$arg if (p_argument==(@-1)) return GetTypeInfo<P@>::get_class_info();
		$
		return PropertyInfo();
	}
#endif
	virtual String get_instance_class() const {
		return T::get_class_static();
	}

	virtual Variant call(Object* p_object,const Variant** p_args,int p_arg_count, Variant::CallError& r_error) {

		T *instance=Object::cast_to<T>(p_object);
		r_error.error=Variant::CallError::CALL_OK;
#ifdef DEBUG_METHODS_ENABLED

		ERR_FAIL_COND_V(!instance,Variant());
		if (p_arg_count>get_argument_count()) {
			r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.argument=get_argument_count();
			return Variant();

		}
		if (p_arg_count<(get_argument_count()-get_default_argument_count())) {

			r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument=get_argument_count()-get_default_argument_count();
			return Variant();
		}
		$arg CHECK_ARG(@);
		$
#endif
		$ifret Variant ret = $(instance->*method)($arg, _VC(@)$);
		$ifret return Variant(ret);$
		$ifnoret return Variant();$
	}

#ifdef PTRCALL_ENABLED
	virtual void ptrcall(Object*p_object,const void** p_args,void *r_ret) {

		T *instance=Object::cast_to<T>(p_object);
		$ifret PtrToArg<R>::encode( $ (instance->*method)($arg, PtrToArg<P@>::convert(p_args[@-1])$) $ifret ,r_ret)$ ;
	}
#endif
	MethodBind$argc$$ifret R$$ifconst C$ () {
#ifdef DEBUG_METHODS_ENABLED
		_set_const($ifconst true$$ifnoconst false$);
		_generate_argument_types($argc$);
#else
		set_argument_count($argc$);
#endif

		$ifret _set_returns(true); $
	};
};

template<class T $ifret ,class R$ $ifargs ,$ $arg, class P@$>
MethodBind* create_method_bind($ifret R$ $ifnoret void$ (T::*p_method)($arg, P@$) $ifconst const$ ) {

	MethodBind$argc$$ifret R$$ifconst C$<T $ifret ,R$ $ifargs ,$ $arg, P@$> * a = memnew( (MethodBind$argc$$ifret R$$ifconst C$<T $ifret ,R$ $ifargs ,$ $arg, P@$>) );
	a->method=p_method;
	return a;
}
#endif
"""

template = """
#ifndef TYPED_METHOD_BIND
$iftempl template<$ $ifret class R$ $ifretargs ,$ $arg, class P@$ $iftempl >$
class MethodBind$argc$$ifret R$$ifconst C$ : public MethodBind {

public:

	StringName type_name;
	$ifret R$ $ifnoret void$ (__UnexistingClass::*method)($arg, P@$) $ifconst const$;

#ifdef DEBUG_METHODS_ENABLED
	virtual Variant::Type _gen_argument_type(int p_arg) const { return _get_argument_type(p_arg); }

	Variant::Type _get_argument_type(int p_argument) const {
		$ifret if (p_argument==-1) return (Variant::Type)GetTypeInfo<R>::VARIANT_TYPE;$
		$arg if (p_argument==(@-1)) return (Variant::Type)GetTypeInfo<P@>::VARIANT_TYPE;
		$
		return Variant::NIL;
	}

	virtual PropertyInfo _gen_argument_type_info(int p_argument) const {
		$ifret if (p_argument==-1) return GetTypeInfo<R>::get_class_info();$
		$arg if (p_argument==(@-1)) return GetTypeInfo<P@>::get_class_info();
		$
		return PropertyInfo();
	}

#endif
	virtual String get_instance_class() const {
		return type_name;
	}

	virtual Variant call(Object* p_object,const Variant** p_args,int p_arg_count, Variant::CallError& r_error) {

		__UnexistingClass *instance = (__UnexistingClass*)p_object;

		r_error.error=Variant::CallError::CALL_OK;
#ifdef DEBUG_METHODS_ENABLED

		ERR_FAIL_COND_V(!instance,Variant());
		if (p_arg_count>get_argument_count()) {
			r_error.error=Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
			r_error.argument=get_argument_count();
			return Variant();
		}

		if (p_arg_count<(get_argument_count()-get_default_argument_count())) {

			r_error.error=Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
			r_error.argument=get_argument_count()-get_default_argument_count();
			return Variant();
		}

		$arg CHECK_ARG(@);
		$
#endif
		$ifret Variant ret = $(instance->*method)($arg, _VC(@)$);
		$ifret return Variant(ret);$
		$ifnoret return Variant();$
	}
#ifdef PTRCALL_ENABLED
	virtual void ptrcall(Object*p_object,const void** p_args,void *r_ret) {
		__UnexistingClass *instance = (__UnexistingClass*)p_object;
		$ifret PtrToArg<R>::encode( $ (instance->*method)($arg, PtrToArg<P@>::convert(p_args[@-1])$) $ifret ,r_ret) $ ;
	}
#endif
	MethodBind$argc$$ifret R$$ifconst C$ () {
#ifdef DEBUG_METHODS_ENABLED
		_set_const($ifconst true$$ifnoconst false$);
		_generate_argument_types($argc$);
#else
		set_argument_count($argc$);
#endif
		$ifret _set_returns(true); $


	};
};

template<class T $ifret ,class R$ $ifargs ,$ $arg, class P@$>
MethodBind* create_method_bind($ifret R$ $ifnoret void$ (T::*p_method)($arg, P@$) $ifconst const$ ) {

	MethodBind$argc$$ifret R$$ifconst C$ $iftempl <$  $ifret R$ $ifretargs ,$ $arg, P@$ $iftempl >$ * a = memnew( (MethodBind$argc$$ifret R$$ifconst C$ $iftempl <$ $ifret R$ $ifretargs ,$ $arg, P@$ $iftempl >$) );
	union {

		$ifret R$ $ifnoret void$ (T::*sm)($arg, P@$) $ifconst const$;
		$ifret R$ $ifnoret void$ (__UnexistingClass::*dm)($arg, P@$) $ifconst const$;
	} u;
	u.sm=p_method;
	a->method=u.dm;
	a->type_name=T::get_class_static();
	return a;
}
#endif
"""


def make_version(template, nargs, argmax, const, ret):

    intext = template
    from_pos = 0
    outtext = ""

    while(True):
        to_pos = intext.find("$", from_pos)
        if (to_pos == -1):
            outtext += intext[from_pos:]
            break
        else:
            outtext += intext[from_pos:to_pos]
        end = intext.find("$", to_pos + 1)
        if (end == -1):
            break  # ignore
        macro = intext[to_pos + 1:end]
        cmd = ""
        data = ""

        if (macro.find(" ") != -1):
            cmd = macro[0:macro.find(" ")]
            data = macro[macro.find(" ") + 1:]
        else:
            cmd = macro

        if (cmd == "argc"):
            outtext += str(nargs)
        if (cmd == "ifret" and ret):
            outtext += data
        if (cmd == "ifargs" and nargs):
            outtext += data
        if (cmd == "ifretargs" and nargs and ret):
            outtext += data
        if (cmd == "ifconst" and const):
            outtext += data
        elif (cmd == "ifnoconst" and not const):
            outtext += data
        elif (cmd == "ifnoret" and not ret):
            outtext += data
        elif (cmd == "iftempl" and (nargs > 0 or ret)):
            outtext += data
        elif (cmd == "arg,"):
            for i in range(1, nargs + 1):
                if (i > 1):
                    outtext += ", "
                outtext += data.replace("@", str(i))
        elif (cmd == "arg"):
            for i in range(1, nargs + 1):
                outtext += data.replace("@", str(i))
        elif (cmd == "noarg"):
            for i in range(nargs + 1, argmax + 1):
                outtext += data.replace("@", str(i))
        elif (cmd == "noarg"):
            for i in range(nargs + 1, argmax + 1):
                outtext += data.replace("@", str(i))

        from_pos = end + 1

    return outtext


def run(target, source, env):

    versions = 11
    versions_ext = 6
    text = ""
    text_ext = ""

    for i in range(0, versions + 1):

        t = ""
        t += make_version(template, i, versions, False, False)
        t += make_version(template_typed, i, versions, False, False)
        t += make_version(template, i, versions, False, True)
        t += make_version(template_typed, i, versions, False, True)
        t += make_version(template, i, versions, True, False)
        t += make_version(template_typed, i, versions, True, False)
        t += make_version(template, i, versions, True, True)
        t += make_version(template_typed, i, versions, True, True)
        if (i >= versions_ext):
            text_ext += t
        else:
            text += t

    f = open(target[0].path, "w")
    f.write(text)
    f.close()

    f = open(target[1].path, "w")
    f.write(text_ext)
    f.close()
