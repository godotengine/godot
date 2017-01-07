/*************************************************************************/
/*  audio_stream_mpc.cpp                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "audio_stream_mpc.h"


Error AudioStreamPlaybackMPC::_open_file() {

	if (f) {
		memdelete(f);
		f=NULL;
	}
	Error err;
	//printf("mpc open file %ls\n", file.c_str());
	f=FileAccess::open(file,FileAccess::READ,&err);

	if (err) {
		f=NULL;
		ERR_FAIL_V(err);
		return err;
	}

	//printf("file size is %i\n", f->get_len());
	//f->seek_end(0);
	streamlen=f->get_len();
	//f->seek(0);
	if (streamlen<=0) {
		memdelete(f);
		f=NULL;
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	data_ofs=0;
	if (preload) {

		data.resize(streamlen);
		PoolVector<uint8_t>::Write w = data.write();
		f->get_buffer(&w[0],streamlen);
		memdelete(f);
		f=NULL;

	}

	return OK;
}

void AudioStreamPlaybackMPC::_close_file() {

	if (f) {
		memdelete(f);
		f=NULL;
	}
	data.resize(0);
	streamlen=0;
	data_ofs=0;
}

int AudioStreamPlaybackMPC::_read_file(void *p_dst,int p_bytes) {

	if (f)
		return f->get_buffer((uint8_t*)p_dst,p_bytes);

	PoolVector<uint8_t>::Read r = data.read();
	if (p_bytes+data_ofs > streamlen) {
		p_bytes=streamlen-data_ofs;
	}

	copymem(p_dst,&r[data_ofs],p_bytes);
	//print_line("read file: "+itos(p_bytes));
	data_ofs+=p_bytes;
	return p_bytes;
}

bool AudioStreamPlaybackMPC::_seek_file(int p_pos){

	if (p_pos<0 || p_pos>streamlen)
		return false;

	if (f) {
		f->seek(p_pos);
		return true;
	}

	//print_line("read file to: "+itos(p_pos));
	data_ofs=p_pos;
	return true;

}
int AudioStreamPlaybackMPC::_tell_file()  const{

	if (f)
		return f->get_pos();

	//print_line("tell file, get: "+itos(data_ofs));
	return data_ofs;

}

int AudioStreamPlaybackMPC::_sizeof_file() const{

	//print_line("sizeof file, get: "+itos(streamlen));
	return streamlen;
}

bool AudioStreamPlaybackMPC::_canseek_file() const{

	//print_line("canseek file, get true");
	return true;
}

/////////////////////

mpc_int32_t AudioStreamPlaybackMPC::_mpc_read(mpc_reader *p_reader,void *p_dst, mpc_int32_t p_bytes) {

	AudioStreamPlaybackMPC *smpc=(AudioStreamPlaybackMPC *)p_reader->data;
	return smpc->_read_file(p_dst,p_bytes);
}

mpc_bool_t AudioStreamPlaybackMPC::_mpc_seek(mpc_reader *p_reader,mpc_int32_t p_offset) {

	AudioStreamPlaybackMPC *smpc=(AudioStreamPlaybackMPC *)p_reader->data;
	return smpc->_seek_file(p_offset);

}
mpc_int32_t AudioStreamPlaybackMPC::_mpc_tell(mpc_reader *p_reader) {

	AudioStreamPlaybackMPC *smpc=(AudioStreamPlaybackMPC *)p_reader->data;
	return smpc->_tell_file();

}
mpc_int32_t AudioStreamPlaybackMPC::_mpc_get_size(mpc_reader *p_reader) {

	AudioStreamPlaybackMPC *smpc=(AudioStreamPlaybackMPC *)p_reader->data;
	return smpc->_sizeof_file();


}
mpc_bool_t AudioStreamPlaybackMPC::_mpc_canseek(mpc_reader *p_reader) {

	AudioStreamPlaybackMPC *smpc=(AudioStreamPlaybackMPC *)p_reader->data;
	return smpc->_canseek_file();
}




int AudioStreamPlaybackMPC::mix(int16_t* p_bufer,int p_frames) {

	if (!active || paused)
		return 0;

	int todo=p_frames;

	while(todo>MPC_DECODER_BUFFER_LENGTH/si.channels) {

		mpc_frame_info frame;

		frame.buffer=sample_buffer;

		mpc_status err = mpc_demux_decode(demux, &frame);
		if (frame.bits!=-1) {

			int16_t *dst_buff = p_bufer;

#ifdef MPC_FIXED_POINT

			for( int i = 0; i < frame.samples * si.channels; i++) {
				int tmp = sample_buffer[i] >> MPC_FIXED_POINT_FRACTPART;
				if (tmp > ((1 << 15) - 1)) tmp = ((1 << 15) - 1);
				if (tmp < -(1 << 15)) tmp = -(1 << 15);
				dst_buff[i] = tmp;
			}
#else
			for( int i = 0; i < frame.samples * si.channels; i++) {

				int tmp = Math::fast_ftoi(sample_buffer[i]*32767.0);
				if (tmp > ((1 << 15) - 1)) tmp = ((1 << 15) - 1);
				if (tmp < -(1 << 15)) tmp = -(1 << 15);
				dst_buff[i] = tmp;

			}

#endif

			int frames = frame.samples;
			p_bufer+=si.channels*frames;
			todo-=frames;
		} else {

			if (err != MPC_STATUS_OK) {

				stop();
				ERR_PRINT("Error decoding MPC");
				break;
			} else {

				//finished
				if (!loop) {
					stop();
					break;
				} else {


					loops++;
					mpc_demux_exit(demux);
					_seek_file(0);
					demux = mpc_demux_init(&reader);
					//do loop somehow

				}
			}
		}
	}

	return p_frames-todo;
}

Error AudioStreamPlaybackMPC::_reload() {

	ERR_FAIL_COND_V(demux!=NULL, ERR_FILE_ALREADY_IN_USE);

	Error err = _open_file();
	ERR_FAIL_COND_V(err!=OK,ERR_CANT_OPEN);

	demux = mpc_demux_init(&reader);
	ERR_FAIL_COND_V(!demux,ERR_CANT_CREATE);
	mpc_demux_get_info(demux,  &si);

	return OK;
}

void AudioStreamPlaybackMPC::set_file(const String& p_file) {

	file=p_file;

	Error err = _open_file();
	ERR_FAIL_COND(err!=OK);
	demux = mpc_demux_init(&reader);
	ERR_FAIL_COND(!demux);
	mpc_demux_get_info(demux,  &si);
	stream_min_size=MPC_DECODER_BUFFER_LENGTH*2/si.channels;
	stream_rate=si.sample_freq;
	stream_channels=si.channels;

	mpc_demux_exit(demux);
	demux=NULL;
	_close_file();

}


String AudioStreamPlaybackMPC::get_file() const {

	return file;
}


void AudioStreamPlaybackMPC::play(float p_offset) {


	if (active)
		stop();
	active=false;

	Error err = _open_file();
	ERR_FAIL_COND(err!=OK);
	if (_reload()!=OK)
		return;
	active=true;
	loops=0;

}

void AudioStreamPlaybackMPC::stop()  {


	if (!active)
		return;
	if (demux) {
		mpc_demux_exit(demux);
		demux=NULL;
	}
	_close_file();
	active=false;

}
bool AudioStreamPlaybackMPC::is_playing() const  {

	return active;
}


void AudioStreamPlaybackMPC::set_loop(bool p_enable)  {

	loop=p_enable;
}
bool AudioStreamPlaybackMPC::has_loop() const  {

	return loop;
}

float AudioStreamPlaybackMPC::get_length() const {

	return 0;
}

String AudioStreamPlaybackMPC::get_stream_name() const {

	return "";
}

int AudioStreamPlaybackMPC::get_loop_count() const {

	return 0;
}

float AudioStreamPlaybackMPC::get_pos() const {

	return 0;
}
void AudioStreamPlaybackMPC::seek_pos(float p_time) {


}


void AudioStreamPlaybackMPC::_bind_methods() {

	ClassDB::bind_method(_MD("set_file","name"),&AudioStreamPlaybackMPC::set_file);
	ClassDB::bind_method(_MD("get_file"),&AudioStreamPlaybackMPC::get_file);

	ADD_PROPERTYNZ( PropertyInfo(Variant::STRING,"file",PROPERTY_HINT_FILE,"mpc"), _SCS("set_file"), _SCS("get_file"));

}

AudioStreamPlaybackMPC::AudioStreamPlaybackMPC() {

	preload=false;
	f=NULL;
	streamlen=0;
	data_ofs=0;
	active=false;
	paused=false;
	loop=false;
	demux=NULL;
	reader.data=this;
	reader.read=_mpc_read;
	reader.seek=_mpc_seek;
	reader.tell=_mpc_tell;
	reader.get_size=_mpc_get_size;
	reader.canseek=_mpc_canseek;
	loops=0;

}

AudioStreamPlaybackMPC::~AudioStreamPlaybackMPC() {

	stop();

	if (f)
		memdelete(f);
}



RES ResourceFormatLoaderAudioStreamMPC::load(const String &p_path, const String& p_original_path, Error *r_error) {
	if (r_error)
		*r_error=OK; //streamed so it will always work..
	AudioStreamMPC *mpc_stream = memnew(AudioStreamMPC);
	mpc_stream->set_file(p_path);
	return Ref<AudioStreamMPC>(mpc_stream);
}

void ResourceFormatLoaderAudioStreamMPC::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("mpc");
}
bool ResourceFormatLoaderAudioStreamMPC::handles_type(const String& p_type) const {

	return (p_type=="AudioStream") || (p_type=="AudioStreamMPC");
}

String ResourceFormatLoaderAudioStreamMPC::get_resource_type(const String &p_path) const {

	if (p_path.extension().to_lower()=="mpc")
		return "AudioStreamMPC";
	return "";
}

