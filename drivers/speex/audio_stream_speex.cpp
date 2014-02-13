#include "audio_stream_speex.h"

#include "os_support.h"
#include "os/os.h"
#define READ_CHUNK 1024

static _FORCE_INLINE_ uint16_t le_short(uint16_t s)
{
   uint16_t ret=s;
#if 0 //def BIG_ENDIAN_ENABLED
   ret =  s>>8;
   ret += s<<8;
#endif
   return ret;
}


void AudioStreamSpeex::update() {

	_THREAD_SAFE_METHOD_;
	//printf("update, loops %i, read ofs %i\n", (int)loops, read_ofs);
	//printf("playing %i, paused %i\n", (int)playing, (int)paused);

	if (!playing || paused || !data.size())
		return;

	/*
	if (read_ofs >= data.size()) {
		if (loops) {
			reload();
			++loop_count;
		} else {
			return;
		};
	};
	*/

	int todo = get_todo();
	if (todo < page_size) {
		return;
	};

	int eos = 0;

	while (todo > page_size) {

		int ret=0;
		while ((todo>page_size && packets_available && !eos) || (ret = ogg_sync_pageout(&oy, &og))==1) {

			if (!packets_available) {
			/*Add page to the bitstream*/
				ogg_stream_pagein(&os, &og);
				page_granule = ogg_page_granulepos(&og);
				page_nb_packets = ogg_page_packets(&og);
				packet_no=0;
				if (page_granule>0 && frame_size)
				{
					skip_samples = page_nb_packets*frame_size*nframes - (page_granule-last_granule);
					if (ogg_page_eos(&og))
						skip_samples = -skip_samples;
					/*else if (!ogg_page_bos(&og))
					skip_samples = 0;*/
				} else
				{
					skip_samples = 0;
				}


				last_granule = page_granule;
				packets_available=true;
			}
			/*Extract all available packets*/
			//int packet_no=0;
			while (todo > page_size && !eos) {

				if (ogg_stream_packetout(&os, &op)!=1) {
					packets_available=false;
					break;
				}

				packet_no++;


				/*End of stream condition*/
				if (op.e_o_s)
					eos=1;

				/*Copy Ogg packet to Speex bitstream*/
				speex_bits_read_from(&bits, (char*)op.packet, op.bytes);


				for (int j=0;j!=nframes;j++)
				{

					int16_t* out = get_write_buffer();

					int ret;
					/*Decode frame*/
					ret = speex_decode_int(st, &bits, out);

					/*for (i=0;i<frame_size*channels;i++)
					  printf ("%d\n", (int)output[i]);*/

					if (ret==-1) {
						printf("decode returned -1\n");
						break;
					};
					if (ret==-2)
					{
						OS::get_singleton()->printerr( "Decoding error: corrupted stream?\n");
						break;
					}
					if (speex_bits_remaining(&bits)<0)
					{
						OS::get_singleton()->printerr( "Decoding overflow: corrupted stream?\n");
						break;
					}
					//if (channels==2)
					//	speex_decode_stereo_int(output, frame_size, &stereo);


					/*Convert to short and save to output file*/
					for (int i=0;i<frame_size*get_channel_count();i++) {
						out[i]=le_short(out[i]);
					}


					{

						int frame_offset = 0;
						int new_frame_size = frame_size;

						/*printf ("packet %d %d\n", packet_no, skip_samples);*/
						if (packet_no == 1 && j==0 && skip_samples > 0)
						{
							/*printf ("chopping first packet\n");*/
							new_frame_size -= skip_samples;
							frame_offset = skip_samples;
						}
						if (packet_no == page_nb_packets && skip_samples < 0)
						{
							int packet_length = nframes*frame_size+skip_samples;
							new_frame_size = packet_length - j*frame_size;
							if (new_frame_size<0)
								new_frame_size = 0;
							if (new_frame_size>frame_size)
								new_frame_size = frame_size;
							/*printf ("chopping end: %d %d %d\n", new_frame_size, packet_length, packet_no);*/
						}


						write(new_frame_size);
						todo-=new_frame_size;
					}
				}

			};
		};
		//todo = get_todo();

		//todo is still greater than page size, can write more
		if (todo > page_size || eos) {
			if (read_ofs < data.size()) {

				//char *buf;
				int nb_read = MIN(data.size() - read_ofs, READ_CHUNK);

				/*Get the ogg buffer for writing*/
				char* ogg_dst = ogg_sync_buffer(&oy, nb_read);
				/*Read bitstream from input file*/
				copymem(ogg_dst, &data[read_ofs], nb_read);
				read_ofs += nb_read;
				ogg_sync_wrote(&oy, nb_read);
			} else {
				if (loops) {					
					reload();
					++loop_count;
				} else {
					playing=false;
					unload();
					break;
				};
			}
		};
	};
};


void AudioStreamSpeex::unload() {

	_THREAD_SAFE_METHOD_

	if (!active) return;

	speex_bits_destroy(&bits);
	if (st)
		speex_decoder_destroy(st);
	active = false;
	//data.resize(0);
	st = NULL;

	frame_size = 0;
	page_size = 0;
	loop_count = 0;
}

void *AudioStreamSpeex::process_header(ogg_packet *op, int *frame_size, int *rate, int *nframes, int *channels, int *extra_headers) {

	void *st;
	SpeexHeader *header;
	int modeID;
	SpeexCallback callback;

	header = speex_packet_to_header((char*)op->packet, op->bytes);
	if (!header)
	{
		OS::get_singleton()->printerr( "Cannot read header\n");
		return NULL;
	}
	if (header->mode >= SPEEX_NB_MODES)
	{
		OS::get_singleton()->printerr( "Mode number %d does not (yet/any longer) exist in this version\n",
				 header->mode);
		return NULL;
	}

	modeID = header->mode;

	const SpeexMode *mode = speex_lib_get_mode (modeID);

	if (header->speex_version_id > 1)
	{
		OS::get_singleton()->printerr( "This file was encoded with Speex bit-stream version %d, which I don't know how to decode\n", header->speex_version_id);
		return NULL;
	}

	if (mode->bitstream_version < header->mode_bitstream_version)
	{
		OS::get_singleton()->printerr( "The file was encoded with a newer version of Speex. You need to upgrade in order to play it.\n");
		return NULL;
	}
	if (mode->bitstream_version > header->mode_bitstream_version)
	{
		OS::get_singleton()->printerr( "The file was encoded with an older version of Speex. You would need to downgrade the version in order to play it.\n");
		return NULL;
	}

	void* state = speex_decoder_init(mode);
	if (!state)
	{
		OS::get_singleton()->printerr( "Decoder initialization failed.\n");
		return NULL;
	}
	//speex_decoder_ctl(state, SPEEX_SET_ENH, &enh_enabled);
	speex_decoder_ctl(state, SPEEX_GET_FRAME_SIZE, frame_size);

	if (!*rate)
		*rate = header->rate;

	speex_decoder_ctl(state, SPEEX_SET_SAMPLING_RATE, rate);

	*nframes = header->frames_per_packet;

	*channels = header->nb_channels;

	if (*channels!=1) {
		OS::get_singleton()->printerr("Only MONO speex streams supported\n");
		return NULL;
	}

	*extra_headers = header->extra_headers;

	speex_free(header);
	return state;
}



void AudioStreamSpeex::reload() {

	_THREAD_SAFE_METHOD_

	if (active)
		unload();

	if (!data.size())
		return;

	ogg_sync_init(&oy);
	speex_bits_init(&bits);

	read_ofs = 0;
//	char *buf;

	int packet_count = 0;
	int extra_headers = 0;
	int stream_init = 0;

	page_granule=0;
	last_granule=0;
	skip_samples=0;
	page_nb_packets=0;
	packets_available=false;
	packet_no=0;

	int eos = 0;

	do {

		/*Get the ogg buffer for writing*/
		int nb_read = MIN(data.size() - read_ofs, READ_CHUNK);
		char* ogg_dst = ogg_sync_buffer(&oy, nb_read);
		/*Read bitstream from input file*/
		copymem(ogg_dst, &data[read_ofs], nb_read);
		read_ofs += nb_read;
		ogg_sync_wrote(&oy, nb_read);

		/*Loop for all complete pages we got (most likely only one)*/
		while (ogg_sync_pageout(&oy, &og)==1) {

			int packet_no;
			if (stream_init == 0) {
				ogg_stream_init(&os, ogg_page_serialno(&og));
				stream_init = 1;
			}
			/*Add page to the bitstream*/
			ogg_stream_pagein(&os, &og);
			page_granule = ogg_page_granulepos(&og);
			page_nb_packets = ogg_page_packets(&og);
			if (page_granule>0 && frame_size)
			{
				skip_samples = page_nb_packets*frame_size*nframes - (page_granule-last_granule);
				if (ogg_page_eos(&og))
					skip_samples = -skip_samples;
				/*else if (!ogg_page_bos(&og))
				  skip_samples = 0;*/
			} else
			{
				skip_samples = 0;
			}


			last_granule = page_granule;
			/*Extract all available packets*/
			packet_no=0;
			while (!eos && ogg_stream_packetout(&os, &op)==1)
			{
				/*If first packet, process as Speex header*/
				if (packet_count==0)
				{
					int rate = 0;
					int channels;
					st = process_header(&op, &frame_size, &rate, &nframes, &channels, &extra_headers);
					if (!nframes)
						nframes=1;
					if (!st) {
						unload();
						return;
					};

					page_size = nframes * frame_size;

					_setup(channels, rate,page_size);

				} else if (packet_count==1)
				{
				} else if (packet_count<=1+extra_headers)
				{
					/* Ignore extra headers */
				};
			};
			++packet_count;
		};

	} while (packet_count <= extra_headers);

	active = true;

}

void AudioStreamSpeex::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_file","file"),&AudioStreamSpeex::set_file);
	ObjectTypeDB::bind_method(_MD("get_file"),&AudioStreamSpeex::get_file);

	ObjectTypeDB::bind_method(_MD("_set_bundled"),&AudioStreamSpeex::_set_bundled);
	ObjectTypeDB::bind_method(_MD("_get_bundled"),&AudioStreamSpeex::_get_bundled);

	ADD_PROPERTY( PropertyInfo(Variant::DICTIONARY,"_bundled",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_BUNDLE),_SCS("_set_bundled"),_SCS("_get_bundled"));
	ADD_PROPERTY( PropertyInfo(Variant::STRING,"file",PROPERTY_HINT_FILE,"*.spx"),_SCS("set_file"),_SCS("get_file"));
};

void AudioStreamSpeex::_set_bundled(const Dictionary& dict) {

	ERR_FAIL_COND( !dict.has("filename"));
	ERR_FAIL_COND( !dict.has("data"));

	filename = dict["filename"];
	data = dict["data"];
};

Dictionary AudioStreamSpeex::_get_bundled() const {

	Dictionary d;
	d["filename"] = filename;
	d["data"] = data;
	return d;
};


String AudioStreamSpeex::get_file() const {

	return filename;
};

void AudioStreamSpeex::set_file(const String& p_file){

	if (filename == p_file)
		return;

	if (active) {
		unload();
	}

	if (p_file == "") {
		data.resize(0);
		return;
	};

	Error err;
	FileAccess* file = FileAccess::open(p_file, FileAccess::READ,&err);
	if (err != OK) {
		data.resize(0);
	};
	ERR_FAIL_COND(err != OK);

	filename = p_file;
	data.resize(file->get_len());
	int read = file->get_buffer(&data[0], data.size());
	memdelete(file);

	reload();
}

void AudioStreamSpeex::play() {

	_THREAD_SAFE_METHOD_

	reload();
	if (!active)
		return;
	playing = true;

}
void AudioStreamSpeex::stop(){

	_THREAD_SAFE_METHOD_
	unload();
	playing = false;
	_clear();
}
bool AudioStreamSpeex::is_playing() const{

	return _is_ready() && (playing || (get_total() - get_todo() -1 > 0));
}

void AudioStreamSpeex::set_paused(bool p_paused){

	playing = !p_paused;
	paused = p_paused;
}
bool AudioStreamSpeex::is_paused(bool p_paused) const{

	return paused;
}

void AudioStreamSpeex::set_loop(bool p_enable){

	loops = p_enable;
}
bool AudioStreamSpeex::has_loop() const{

	return loops;
}

float AudioStreamSpeex::get_length() const{

	return 0;
}

String AudioStreamSpeex::get_stream_name() const{

	return "";
}

int AudioStreamSpeex::get_loop_count() const{

	return 0;
}

float AudioStreamSpeex::get_pos() const{

	return 0;
}
void AudioStreamSpeex::seek_pos(float p_time){


};

bool AudioStreamSpeex::_can_mix() const {

	//return playing;
	return data.size() != 0;
};


AudioStream::UpdateMode AudioStreamSpeex::get_update_mode() const {

	return UPDATE_THREAD;
}

AudioStreamSpeex::AudioStreamSpeex() {

	active=false;
	st = NULL;
}

AudioStreamSpeex::~AudioStreamSpeex() {

	unload();
}

RES ResourceFormatLoaderAudioStreamSpeex::load(const String &p_path,const String& p_original_path) {

	AudioStreamSpeex *stream = memnew(AudioStreamSpeex);
	stream->set_file(p_path);
	return Ref<AudioStreamSpeex>(stream);
}

void ResourceFormatLoaderAudioStreamSpeex::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("spx");
}
bool ResourceFormatLoaderAudioStreamSpeex::handles_type(const String& p_type) const {

	return (p_type=="AudioStream" || p_type=="AudioStreamSpeex");
}

String ResourceFormatLoaderAudioStreamSpeex::get_resource_type(const String &p_path) const {

	if (p_path.extension().to_lower()=="spx")
		return "AudioStreamSpeex";
	return "";
}
