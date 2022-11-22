#ifndef RECORD_H
#define RECORD_H

#include <string>

#include "modules/hub/hub.h"
#include "core/string_name.h"
#include "core/resource.h"
#include "core/script_language.h"
#include "core/vector.h"
#include "core/array.h"

class RawRecord;
class Record;
class RecordEncoder;
class RecordDecoder;
class RecordMask;
class RecordPrimitiveEncoder;
class RecordPrimitiveDecoder;
// class RecordAdvancedEncoder;
// class RecordAdvancedDecoder;


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

	void pre_serialization();
	void post_serialization();
	void pre_deserialization();
	void post_deserialization();
public:
	Record();
	~Record();

	friend class RecordEncoder;
	friend class RecordDecoder;

	_FORCE_INLINE_ void set_track_list(const Vector<StringName>& new_list) { track_list = new_list; }
	_FORCE_INLINE_ Vector<StringName> get_track_list() const { return track_list; }

	_FORCE_INLINE_ void set_encoder(const Ref<RecordEncoder>& new_encoder) { if (new_encoder.is_valid()) encoder = new_encoder; }
	_FORCE_INLINE_ Ref<RecordEncoder> get_encoder() const { return encoder; }

	_FORCE_INLINE_ void set_decoder(const Ref<RecordDecoder>& new_decoder) { if (new_decoder.is_valid()) decoder = new_decoder; }
	_FORCE_INLINE_ Ref<RecordDecoder> get_decoder() const { return decoder; }

	_FORCE_INLINE_ void set_mask(const Ref<RecordMask>& new_mask) { mask = new_mask; }
	_FORCE_INLINE_ Ref<RecordMask> get_mask() const { return mask; }

	Variant read(const StringName& what);
	bool write(const StringName& what, const Variant& with);

	Variant get_track(const StringName& track_name) const;
	Array get_all_tracks() const;

	Variant encode() const;
	bool decode(const Variant& from);
};

class RecordEncoder : public Reference {
	GDCLASS(RecordEncoder, Reference);
protected:
	static void _bind_methods();

	static _FORCE_INLINE_ void print_msg(const char* msg) { Hub::get_singleton()->print_custom("RecordEncoder", msg); }
	static _FORCE_INLINE_ void print_msg(const std::string& msg) { Hub::get_singleton()->print_custom("RecordEncoder", msg.c_str()); }
public:
	RecordEncoder();
	~RecordEncoder();

	Variant encode(Ref<Record> from, const Variant& instruction = Variant()) const;
	virtual Variant builtin_encode(const Ref<Record>& from, const Variant& instruction) const;
};

class RecordDecoder : public Reference {
	GDCLASS(RecordDecoder, Reference);
protected:
	static void _bind_methods();

	static _FORCE_INLINE_ void print_msg(const char* msg) { Hub::get_singleton()->print_custom("RecordDecoder", msg); }
	static _FORCE_INLINE_ void print_msg(const std::string& msg) { Hub::get_singleton()->print_custom("RecordDecoder", msg.c_str()); }
public:
	RecordDecoder();
	~RecordDecoder();

	bool decode(const Variant& from, Ref<Record> to, const Variant& instruction = Variant());
	virtual bool builtin_decode(const Variant& from, Ref<Record> to, const Variant& instruction);
};

class RecordMask : public Reference {
	GDCLASS(RecordMask, Reference);
protected:
	static void _bind_methods();
public:
	RecordMask();
	~RecordMask();

	Variant read(const Ref<Record>& from, const StringName& what);
	bool write(Ref<Record> to, const StringName& what, const Variant& with);

	virtual Variant builtin_read(const Ref<Record>& from, const StringName& what) const;
	virtual bool builtin_write(Ref<Record> to, const StringName& what, const Variant& with);
};

class RecordPrimitiveEncoder : public RecordEncoder{
	GDCLASS(RecordPrimitiveEncoder, RecordEncoder);
protected:
	// static void _bind_methods();
public:
	RecordPrimitiveEncoder() = default;
	~RecordPrimitiveEncoder() = default;

	Variant builtin_encode(const Ref<Record>& from, const Variant& instruction) const override;
};

class RecordPrimitiveDecoder : public RecordDecoder{
	GDCLASS(RecordPrimitiveDecoder, RecordDecoder);
protected:
	// static void _bind_methods();
public:
	RecordPrimitiveDecoder() = default;
	~RecordPrimitiveDecoder() = default;

	bool builtin_decode(const Variant& from, Ref<Record> to, const Variant& instruction) override;
};

enum RecordDependencyType {
	EMBEDDED,
	EXTERNAL,
};

class RawRecordData : public Reference {
	GDCLASS(RawRecordData, Reference);
private:
	uint64_t allocated_id = 0;
public:
	Vector<Pair<StringName, Variant>> table;
	Vector<uint64_t> external_refs;
	StringName extra;
	StringName name;
	RecordDependencyType dependency_type = RecordDependencyType::EMBEDDED;

	uint64_t unique_id = 0;

	friend class RawRecord;
};

class RawRecord : public Reference {
	GDCLASS(RawRecord, Reference);
private:
	uint64_t allocated_id = 0;
	RWLock lock;

	Map<uint64_t, Ref<RawRecordData>> data_table;

	// void internal_encode(const StringName& key, const Variant& val, Ref<RawRecordData>& rdata = Ref<RawRecordData>());
public:
	RawRecord();
	~RawRecord();

	uint64_t create_dependency();
	_FORCE_INLINE_ bool is_dependency_available(const uint64_t& dep_id) const { return data_table.has(dep_id); }	

	void entail_dependency(const uint64_t& internal_id, const Ref<RawRecordData>& data);

	uint64_t iid2uid(const uint64_t& internal_id, const StringName& extra = StringName(""));
	uint64_t uid2iid(const uint64_t& unique_id, const StringName& extra = StringName(""));
	uint64_t xid2xid(const uint64_t& x_id, const bool& reversed, const StringName& extra);

	void clear();
	// void auto_encode(const Map<StringName, Variant>& properties);
};

// class RecordAdvancedEncoder : public RecordEncoder{
// 	GDCLASS(RecordAdvancedEncoder, RecordEncoder);
// private:
// public:
// 	RecordAdvancedEncoder() = default;
// 	~RecordAdvancedEncoder() = default;

// 	Variant builtin_encode(const Ref<Record>& from, const Variant& instruction) const override;
// 	static Ref<RawRecord> internal_encode(const Ref<Record>& from, const Variant& instruction, Ref<RawRecord>& rrec = Ref<RawRecord>(), Ref<RawRecordData>& rdata = Ref<RawRecordData>());
// 	static Ref<RawRecord> internal_encode_res(const RES& from, const Variant& instruction, Ref<RawRecord>& rrec = Ref<RawRecord>(), Ref<RawRecordData>& rdata = Ref<RawRecordData>());

// 	static void recursive_encode(Ref<RawRecord>& raw_record, Ref<RawRecordData>& rec_data, const StringName& track_name, const Variant& track_data);
// 	// static void recursive_encode_primitive(const Ref<RawRecord>& raw_record, const StringName& track_name, const Variant& track_data);
// 	// static void recursive_encode_object(const Ref<RawRecord>& raw_record, const StringName& track_name, const Object* track_data);
// };
// class RecordAdvancedDecoder : public RecordDecoder{
// 	GDCLASS(RecordAdvancedDecoder, RecordDecoder);
// public:
// 	RecordAdvancedDecoder() = default;
// 	~RecordAdvancedDecoder() = default;

// 	bool builtin_decode(const Variant& from, Ref<Record> to, const Variant& instruction) override;
// };

#endif
