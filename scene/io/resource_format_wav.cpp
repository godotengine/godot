/*************************************************************************/
/*  resource_format_wav.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#if 0
#include "resource_format_wav.h"
#include "os/file_access.h"
#include "scene/resources/sample.h"


RES ResourceFormatLoaderWAV::load(const String &p_path, const String& p_original_path, Error *r_error) {
	if (r_error)
		*r_error=ERR_FILE_CANT_OPEN;

	Error err;
	FileAccess *file=FileAccess::open(p_path, FileAccess::READ,&err);

	ERR_FAIL_COND_V( err!=OK, RES() );

	if (r_error)
		*r_error=ERR_FILE_CORRUPT;

	/* CHECK RIFF */
	char riff[5];
	riff[4]=0;
	file->get_buffer((uint8_t*)&riff,4); //RIFF

	if (riff[0]!='R' || riff[1]!='I' || riff[2]!='F' || riff[3]!='F') {

		file->close();
		memdelete(file);
		ERR_FAIL_V( RES() );
	}


	/* GET FILESIZE */
	uint32_t filesize=file->get_32();

	/* CHECK WAVE */

	char wave[4];

	file->get_buffer((uint8_t*)&wave,4); //RIFF

	if (wave[0]!='W' || wave[1]!='A' || wave[2]!='V' || wave[3]!='E') {


		file->close();
		memdelete(file);
		ERR_EXPLAIN("Not a WAV file (no WAVE RIFF Header)")
		ERR_FAIL_V( RES() );
	}

	bool format_found=false;
	bool data_found=false;
	int format_bits=0;
	int format_channels=0;
	int format_freq=0;
	Sample::LoopFormat loop=Sample::LOOP_NONE;
	int loop_begin=0;
	int loop_end=0;


	Ref<Sample> sample( memnew( Sample ) );


	while (!file->eof_reached()) {


		/* chunk */
		char chunkID[4];
		file->get_buffer((uint8_t*)&chunkID,4); //RIFF

		/* chunk size */
		uint32_t chunksize=file->get_32();
		uint32_t file_pos=file->get_pos(); //save file pos, so we can skip to next chunk safely

		if (file->eof_reached()) {

			//ERR_PRINT("EOF REACH");
			break;
		}

		if (chunkID[0]=='f' && chunkID[1]=='m' && chunkID[2]=='t' && chunkID[3]==' ' && !format_found) {
			/* IS FORMAT CHUNK */

			uint16_t compression_code=file->get_16();


			if (compression_code!=1) {
				ERR_PRINT("Format not supported for WAVE file (not PCM). Save WAVE files as uncompressed PCM instead.");
				break;
			}

			format_channels=file->get_16();
			if (format_channels!=1 && format_channels !=2) {

				ERR_PRINT("Format not supported for WAVE file (not stereo or mono)");
				break;

			}

			format_freq=file->get_32(); //sampling rate

			file->get_32(); // average bits/second (unused)
			file->get_16(); // block align (unused)
			format_bits=file->get_16(); // bits per sample

			if (format_bits%8) {

				ERR_PRINT("Strange number of bits in sample (not 8,16,24,32)");
				break;
			}

			/* Don't need anything else, continue */
			format_found=true;
		}


		if (chunkID[0]=='d' && chunkID[1]=='a' && chunkID[2]=='t' && chunkID[3]=='a' && !data_found) {
			/* IS FORMAT CHUNK */
			data_found=true;

			if (!format_found) {
				ERR_PRINT("'data' chunk before 'format' chunk found.");
				break;

			}

			int frames=chunksize;

			frames/=format_channels;
			frames/=(format_bits>>3);

			/*print_line("chunksize: "+itos(chunksize));
			print_line("channels: "+itos(format_channels));
			print_line("bits: "+itos(format_bits));
*/
			sample->create(
					(format_bits==8) ? Sample::FORMAT_PCM8 : Sample::FORMAT_PCM16,
					(format_channels==2)?true:false,
					frames );
			sample->set_mix_rate( format_freq );

			int len=frames;
			if (format_channels==2)
				len*=2;
			if (format_bits>8)
				len*=2;

			PoolVector<uint8_t> data;
			data.resize(len);
			PoolVector<uint8_t>::Write dataw = data.write();
			void * data_ptr = dataw.ptr();

			for (int i=0;i<frames;i++) {


				for (int c=0;c<format_channels;c++) {


					if (format_bits==8) {
						// 8 bit samples are UNSIGNED

						uint8_t s = file->get_8();
						s-=128;
						int8_t *sp=(int8_t*)&s;

						int8_t *data_ptr8=&((int8_t*)data_ptr)[i*format_channels+c];

						*data_ptr8=*sp;

					} else {
						//16+ bits samples are SIGNED
						// if sample is > 16 bits, just read extra bytes

						uint32_t data=0;
						for (int b=0;b<(format_bits>>3);b++) {

							data|=((uint32_t)file->get_8())<<(b*8);
						}
						data<<=(32-format_bits);


						int32_t s=data;

						int16_t *data_ptr16=&((int16_t*)data_ptr)[i*format_channels+c];

						*data_ptr16=s>>16;
					}
				}

			}

			dataw=PoolVector<uint8_t>::Write();

			sample->set_data(data);


			if (file->eof_reached()) {
				file->close();
				memdelete(file);
				ERR_EXPLAIN("Premature end of file.");
				ERR_FAIL_V(RES());
			}
		}

		if (chunkID[0]=='s' && chunkID[1]=='m' && chunkID[2]=='p' && chunkID[3]=='l') {
			//loop point info!

			for(int i=0;i<10;i++)
				file->get_32(); // i wish to know why should i do this... no doc!

			loop=file->get_32()?Sample::LOOP_PING_PONG:Sample::LOOP_FORWARD;
			loop_begin=file->get_32();
			loop_end=file->get_32();

		}
		file->seek( file_pos+chunksize );
	}

	sample->set_loop_format(loop);
	sample->set_loop_begin(loop_begin);
	sample->set_loop_end(loop_end);

	file->close();
	memdelete(file);

	if (r_error)
		*r_error=OK;


	return sample;

}
void ResourceFormatLoaderWAV::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("wav");
}
bool ResourceFormatLoaderWAV::handles_type(const String& p_type) const {

	return (p_type=="Sample");
}

String ResourceFormatLoaderWAV::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower()=="wav")
		return "Sample";
	return "";
}

#endif
