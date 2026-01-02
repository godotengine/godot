import methods


def run(target, source, env):
    buffer = methods.get_buffer(str(source[0]))
    decomp_size = len(buffer)
    buffer = methods.compress_buffer(buffer)

    with methods.generated_wrapper(str(target[0])) as file:
        file.write(f"""\
#ifdef TOOLS_ENABLED

#include "core/io/compression.h"
#include "core/io/file_access.h"
#include "core/string/ustring.h"

inline constexpr int _gdextension_interface_data_compressed_size = {len(buffer)};
inline constexpr int _gdextension_interface_data_uncompressed_size = {decomp_size};
inline constexpr unsigned char _gdextension_interface_data_compressed[] = {{
	{methods.format_buffer(buffer, 1)}
}};

class GDExtensionInterfaceDump {{
	public:
		static Vector<uint8_t> load_gdextension_interface_file() {{
			Vector<uint8_t> data;
			data.resize(_gdextension_interface_data_uncompressed_size);
			int ret = Compression::decompress(data.ptrw(), _gdextension_interface_data_uncompressed_size, _gdextension_interface_data_compressed, _gdextension_interface_data_compressed_size, Compression::MODE_DEFLATE);
			ERR_FAIL_COND_V_MSG(ret == -1, Vector<uint8_t>(), "Compressed file is corrupt.");
			return data;
		}}

		static void generate_gdextension_interface_file(const String &p_path) {{
			Ref<FileAccess> fa = FileAccess::open(p_path, FileAccess::WRITE);
			ERR_FAIL_COND_MSG(fa.is_null(), vformat("Cannot open file '%s' for writing.", p_path));
			Vector<uint8_t> data = load_gdextension_interface_file();
			if (data.size() > 0) {{
				fa->store_buffer(data.ptr(), data.size());
			}}
		}}
}};

#endif // TOOLS_ENABLED
""")
