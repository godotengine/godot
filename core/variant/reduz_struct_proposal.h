
#if 0

// Implementation in Array

// ContainerTypeValidate needs to be changed:

struct ContainerTypeValidate {
	Variant::Type type = Variant::NIL;
	StringName class_name;
	Ref<Script> script;
	LocalVector<ContainerTypeValidate> struct_members; // Added this for structs, assignment from structs with same layout but different member names should be allowed (because it is likely too difficult to prevent)
	const char *where = "container";
};

// ArrayPrivate needs to be changed:

class ArrayPrivate {
public:
	SafeRefCount refcount;
	Vector<Variant> array;
	Variant *read_only = nullptr; // If enabled, a pointer is used to a temporary value that is used to return read-only values.
	ContainerTypeValidate typed;

	// Added struct stuff:
	uint32_t struct_size = 0;
	StringName * struct_member_names = nullptr;
	bool struct_array = false;

	_FORCE_INLINE_ bool is_struct() const {
		return struct_size > 0;
	}

	_FORCE_INLINE_ bool is_struct_array() const {
		return struct_size > 0;
	}

	_FORCE_INLINE_ int32_t find_member_index(const StringName& p_member) const {
		for(uint32_t i = 0 ; i<struct_size ; i++) {
			if (p_member == struct_member_names[i]) {
				return (int32_t)i;
			}
		}

		return -1;
	}

	_FORCE_INLINE_ bool validate_member(uint32_t p_index,const Variant& p_value) {
		// needs to check with ContainerValidate, return true is valid
	}

};

// Not using LocalVector and resorting to manual memory allocation to improve on resoure usage and performance.

// Then, besides all the type comparison and checking (leave this to someone else to do)
// Array needs to implement set and get named functions:


Variant Array::get_named(const StringName& p_member) const {
	ERR_FAIL_COND_V(!_p->is_struct(),Variant();
	int32_t offset = _p->find_member_index(p_member);
	ERR_FAIL_INDEX_V(offset,_p->array.size(),Variant());
	return _p->array[offset];
}

void Array::set_named(const StringName& p_member,const Variant& p_value) {
	ERR_FAIL_COND(!_p->is_struct());
	int32_t offset = _p->find_member_index(p_member);
	ERR_FAIL_INDEX(offset,_p->array.size());
	ERR_FAIL_COND(!p->validate_member(p_value);
	_p->array[offset].write[offset]=p_value;
}

// These can be exposed in Variant binder so they support named indexing
// Keep in mind some extra versions with validation that return invalid set/get will need to be added for GDScript to properly throw errors

// Additionally, the Array::set needs to also perform validation if this is a struct.


// FLATTENED ARRAYTS
// We may also want to have a flattneed array, as described before, the goal is when users needs to store data for a huge amount of elements (like lots of bullets) doing
// so in flat memory fashion is a lot more efficient cache wise. Keep in mind that because variants re always 24 bytes in size, there will always be some
// memory wasting, specially if you use many floats. Additionally larger types like Transform3D are allocated separately because they don't
// fit in a Variant, but they have their own memory pools where they will most likely be allocated contiguously too.
// To sump up, this is not as fast as using C structs memory wise, but still orders of magnitude faster and more efficient than using regular arrays.

var a = FlatArray[SomeStruct]
		a.resize(55) //
		print(a.size()) // 1 for single structs
		a[5].member = 819

		// So how this last thing work?
		// The idea is to add a member to the Array class (not ArrayPrivate):

		class Array {
	mutable ArrayPrivate *_p;
	void _unref() const;
	uint32_t struct_offset = 0; // Add this
public:

	// And the functions described above actually are implemented like this:

	Variant Array::get_named(const StringName& p_member) const {
	ERR_FAIL_COND_V(!_p->struct_layout.is_struct(),Variant();
	int32_t offset = _p->find_member_index(p_member);
	offset += struct_offset * _p->struct_size;
	ERR_FAIL_INDEX_V(offset,_p->array.size(),Variant());
	return _p->array[offset];
	}

	void Array::set_named(const StringName& p_member,const Variant& p_value) {
	ERR_FAIL_COND(!_p->struct_layout.is_struct());
	int32_t offset = _p->find_member_index(p_member);
	ERR_FAIL_COND(!p->validate_member(p_value);
	offset += struct_offset * _p->struct_size;
	ERR_FAIL_INDEX(offset,_p->array.size());
	_p->array[offset].write[offset]=p_value;
	}

	Array Array::struct_at(int p_index) const {
	ERR_FAIL_COND_V(!_p->struct_layout.is_struct(),Array());
	ERR_FAIL_INDEX_V(p_index,_p->array.size() / _p->struct_layout.get_member_count(),Array())
	Array copy = *this;
	copy.struct_offset = p_index;
	return copy;
	}

	// Of course, functions such as size, resize, push_back, etc. in the array should not be modified in Array itself, as this makes serialization of arrays
	impossible at the low level.
			// These functions should be special cased with special versions in Variant::call, including ther operator[] to return struct_at internally if in flattened array mode.
			// Iteration of flattened arrays (when type information is known) could be done extremely efficiently by the GDScript VM by simply increasing the offset variable in each loop. Additionally, the GDScript VM, being typed, could be simply instructed to get members by offset, and hence it could use functions like this:

			Variant Array::get_struct_member_by_offset(uint32_t p_offset) const {
	ERR_FAIL_COND_V(!_p->struct_layout.is_struct(),Variant();
	int32_t offset = p_offset;
	offset += struct_offset * _p->struct_size;
	ERR_FAIL_INDEX_V(offset,_p->array.size(),Variant());
	return _p->array[offset];
	}

	void Array::set_struct_member_by_offset(uint32_t p_offset,const Variant& p_value) {
	ERR_FAIL_COND(!_p->struct_layout.is_struct());
	int32_t offset = p_offset;
	offset += struct_offset * _p->struct_size;
	ERR_FAIL_INDEX(offset,_p->array.size());
	_p->array[offset].write[offset]=p_value;
	}


	// TYPE DESCRIPTIONS in C++

	// Another problem we will face with this approach is that there are many cases where we will want to actually describe the type.
	// If we had a function that returned a dictionary and now we want to change it to a struct because its easier for the user to use (description in doc, autocomplete in GDScript, etc) we must find a way. As an example for typed arrays we have:

	TypedArray<Type> get_someting() const;

	// And the binder takes care. Ideally we want to be able to do something like:

	Struct<StructLayout> get_someting() const;

	// We know we want to eventually do things like like this exposed to the binder.

	TypedArray<Struct<PropertyInfoLayout>> get_property_list();

	// So what are Struct and StructLayout?

	//We would like to do PropertyInfoLayout like this:


	STRUCT_LAYOUT( ProperyInfo, STRUCT_MEMBER("name", Variant::STRING), STRUCT_MEMBER("type", Variant::INT), STRUCT_MEMBER("hint", Variant::INT), STRUCT_MEMBER("hint_string", Variant::STRING), STRUCT_MEMBER("class_name", Variant::STRING) );

	// How does this convert to C?

	// Here is a rough sketch
	struct StructMember {
	StringName name;
	Variant:Type type;
	StringName class_name;
	Variant default_value;

	StructMember(const StringName& p_name, const Variant::Type p_type,const Variant& p_default_value = Variant(), const StringName& p_class_name = StringName()) { name = p_name; type=p_type; default_value = p_default_value; class_name = p_class_name; }
	};

// Important so we force SNAME to it, otherwise this will be leaked memory
#define STRUCT_MEMBER(m_name, m_type, m_default_value) StructMember(SNAME(m_name), m_type, m_default_value)
#define STRUCT_CLASS_MEMBER(m_name, m_class) StructMember(SNAME(m_name), Variant::OBJECT, Variant(), m_class)


	// StructLayout should ideally be something that we can define like

#define STRUCT_LAYOUT(m_class, m_name, ...)                                     \
	struct m_name {                                                             \
		_FORCE_INLINE_ static StringName get_class() { return SNAME(#m_class)); \
		}
	_FORCE_INLINE_ static  StringName get_name() { return SNAME(#m_name)); }
	static constexpr uint32_t member_count = GET_ARGUMENT_COUNT;\
	_FORCE_INLINE_ static const StructMember& get_member(uint32_t p_index) {\
	CRASH_BAD_INDEX(p_index,member_count)\
	static StructMember members[member_count]={ __VA_ARGS__ };\
	return members[p_index];\
	}\
};

// Note GET_ARGUMENT_COUNT is a macro that we probably need to add tp typedefs.h, see:
// https://stackoverflow.com/questions/2124339/c-preprocessor-va-args-number-of-arguments

// Okay, so what is Struct<> ?

// Its a similar class to TypedArray


template <class T>
class Struct : public Array {
public:
	typedef Type T;

	_FORCE_INLINE_ void operator=(const Array &p_array) {
	ERR_FAIL_COND_MSG(!is_same_typed(p_array), "Cannot assign a Struct from array with a different format.");
	_ref(p_array);
	}
	_FORCE_INLINE_ Struct(const Variant &p_variant) :
			Array(T::member_count, T::get_member,Array(p_variant)) {
	}
	_FORCE_INLINE_ Struct(const Array &p_array) :
			Array(T::member_count, T::get_member,p_array) {
	}
	_FORCE_INLINE_ Struct() {
		Array(T::member_count, T::get_member) {
		}
	};

	// You likely saw correctly, we pass pointer to T::get_member. This is because we can't pass a structure and we want to initialize ArrayPrivate efficiently without allocating extra memory than needed, plus we want to keep this function around for validation:

	Array::Array(uint32_t p_member_count, const StructMember& (*p_get_member)(uint32_t));
	Array::Array(uint32_t p_member_count, const StructMember& (*p_get_member)(uint32_t),const Array &p_from); // separate one is best for performance since Array() does internal memory allocation when constructed.

// Keep in mind also that GDScript VM is not able to pass a function pointer since this is dynamic, so it will need a separate constructor to initialize the array format. Same reason why the function pointer should not be kept inside of Array.
// Likewise, GDScript may also need to pass a Script for class name, which is what ContainerTypeValidate neeeds.

// Registering the struct to Class DB
// call this inside _bind_methods of the relevant class

// goes in object.h
#define BIND_STRUCT(m_name) ClasDB::register_struct(m_name::get_class(), m_name::get_name(), m_name::member_count, m_name::get_member);

	Then you will also have to add this function `Array ClassDB::instantiate_struct(const StringName &p_class, const StringName& p_struct);` in order to construct them on demand.

			// Optimizations:

			// The idea here is that if GDScript code is typed, it should be able to access everything without any kind of validation or even copies. I will add this in the GDScript optimization proposal I have soon (pointer addressing mode).

			// That said, I think we should consider changing ArrayPrivate::Array from Vector to LocalVector, this should enormously improve performance when accessing untyped (And eventually typed) arrays in GDScript. Arrays are shared, so there is not much of a need to use Vector<> here.

#endif