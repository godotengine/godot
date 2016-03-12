#include "editor_initialize_ssl.h"
#include "certs_compressed.h"
#include "io/stream_peer_ssl.h"
#include "io/compression.h"

void editor_initialize_certificates() {


	ByteArray data;
	data.resize(_certs_uncompressed_size);
	{
		ByteArray::Write w = data.write();
		Compression::decompress(w.ptr(),_certs_uncompressed_size,_certs_compressed,_certs_compressed_size,Compression::MODE_DEFLATE);
	}

	StreamPeerSSL::load_certs_from_memory(data);


}


