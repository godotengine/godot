def generate_compressed_config(config_src, output_dir):
    import os.path
    from compat import byte_to_str

    # Source file
    with open(os.path.join(output_dir, "android_mono_config.gen.cpp"), "w") as cpp:
        with open(config_src, "rb") as f:
            buf = f.read()
            decompr_size = len(buf)
            import zlib

            buf = zlib.compress(buf)
            compr_size = len(buf)

            bytes_seq_str = ""
            for i, buf_idx in enumerate(range(compr_size)):
                if i > 0:
                    bytes_seq_str += ", "
                bytes_seq_str += byte_to_str(buf[buf_idx])

            cpp.write(
                """/* THIS FILE IS GENERATED DO NOT EDIT */
#include "android_mono_config.h"

#ifdef ANDROID_ENABLED

#include "core/io/compression.h"
#include "core/pool_vector.h"

namespace {

// config
static const int config_compressed_size = %d;
static const int config_uncompressed_size = %d;
static const unsigned char config_compressed_data[] = { %s };

} // namespace

String get_godot_android_mono_config() {
	PoolVector<uint8_t> data;
	data.resize(config_uncompressed_size);
	PoolVector<uint8_t>::Write w = data.write();
	Compression::decompress(w.ptr(), config_uncompressed_size, config_compressed_data,
			config_compressed_size, Compression::MODE_DEFLATE);
	String s;
	if (s.parse_utf8((const char *)w.ptr(), data.size())) {
		ERR_FAIL_V(String());
	}
	return s;
}

#endif // ANDROID_ENABLED
"""
                % (compr_size, decompr_size, bytes_seq_str)
            )
