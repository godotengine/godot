#ifndef RECORD_H
#define RECORD_H

#include <string>

#include "modules/hub/hub.h"
#include "core/string_name.h"
#include "core/resource.h"
#include "core/script_language.h"
#include "core/vector.h"
#include "core/array.h"

class Record;
class RecordEncoder;
class RecordDecoder;
class RecordMask;
// class RecordPrimitiveEncoder;
// class RecordPrimitiveDecoder;
// class RecordStandardEncoder;
// class RecordStandardDecoder;
// class RecordResourceLoader;
// class RecordResourceSaver;

class Record : public Resource {
	GDCLASS(Record, Resource);
private:
	Vector<StringName> track_list;

	Ref<RecordEncoder> encoder;
	Ref<RecordDecoder> decoder;
	Ref<RecordMask> mask;
protected:
	static void _bind_methods();
	static _FORCE_INLINE_ void print_msg(const char* msg) { Hub::get_singleton()->print_custom("Record", msg); }
	static _FORCE_INLINE_ void print_msg(const std::string& msg) { Hub::get_singleton()->print_custom("Record", msg.c_str()); }
public:
	Record();
	~Record();

	_FORCE_INLINE_ void set_track_list(const Vector<StringName>& new_list) { track_list = new_list; }
	_FORCE_INLINE_ Vector<StringName> get_track_list() const { return track_list; }

	_FORCE_INLINE_ void set_encoder(const Ref<RecordEncoder>& new_encoder) { if (encoder.is_null()) encoder = new_encoder; }
	_FORCE_INLINE_ Ref<RecordEncoder> get_encoder() const { return encoder; }

	_FORCE_INLINE_ void set_decoder(const Ref<RecordDecoder>& new_decoder) { if (decoder.is_null()) decoder = new_decoder; }
	_FORCE_INLINE_ Ref<RecordDecoder> get_decoder() const { return decoder; }

	_FORCE_INLINE_ void set_mask(const Ref<RecordMask>& new_mask) { mask = new_mask; }
	_FORCE_INLINE_ Ref<RecordMask> get_mask() const { return mask; }

	Variant read(const StringName& what);
	bool write(const StringName& what, const Variant& with);

	Variant get_track(const StringName& track_name) const;
	Array get_all_tracks() const;

	Variant encode() const;
	bool decode(const Ref<Record>& from);
};

class RecordEncoder : public Resource {
	GDCLASS(RecordEncoder, Reference);
protected:
	static void _bind_methods();
public:
	RecordEncoder();
	~RecordEncoder();

	Variant encode(const Ref<Record>& from, const Variant& instruction = Variant()) const;
	virtual Variant builtin_encode(const Ref<Record>& from, const Variant& instruction) const;
};

class RecordDecoder : public Resource {
	GDCLASS(RecordDecoder, Reference);
protected:
	static void _bind_methods();
public:
	RecordDecoder();
	~RecordDecoder();

	bool decode(const Ref<Record>& from, const Ref<Record>& to, const Variant& instruction = Variant());
	virtual bool builtin_decode(const Ref<Record>& from, const Ref<Record>& to, const Variant& instruction);
};

class RecordMask : public Resource {
	GDCLASS(RecordMask, Reference);
protected:
	static void _bind_methods();
public:
	RecordMask();
	~RecordMask();

	Variant read(const Ref<Record>& from, const StringName& what);
	bool write(const Ref<Record>& to, const StringName& what, const Variant& with);

	virtual Variant builtin_read(const Ref<Record>& from, const StringName& what) const;
	virtual bool builtin_write(Ref<Record> to, const StringName& what, const Variant& with);
};

#endif
