#include "core/object.h"

class FileAccess;

class PCKPacker : public Object {

	OBJ_TYPE(PCKPacker, Object);

	FileAccess* file;
	int alignment;

	static void _bind_methods();

	struct File {

		String path;
		String src_path;
		int size;
		uint64_t offset_offset;
	};
	Vector<File> files;

public:
	Error pck_start(const String& p_file, int p_alignment);
	Error add_file(const String& p_file, const String& p_src);
	Error flush(bool p_verbose = false);


	PCKPacker();
	~PCKPacker();
};
