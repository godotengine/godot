// basisu_tool.cpp
// Copyright (C) 2019 Binomial LLC. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "transcoder/basisu.h"
#include "transcoder/basisu_transcoder_internal.h"
#include "basisu_enc.h"
#include "basisu_etc.h"
#include "basisu_gpu_texture.h"
#include "basisu_frontend.h"
#include "basisu_backend.h"
#include "transcoder/basisu_global_selector_palette.h"
#include "basisu_comp.h"
#include "transcoder/basisu_transcoder.h"
#include "basisu_ssim.h"

#define BASISU_CATCH_EXCEPTIONS 1

using namespace basisu;

#define BASISU_TOOL_VERSION "1.10.00"

enum tool_mode
{
	cDefault,
	cCompress,
	cValidate,
	cUnpack,
	cCompare,
	cVersion,
};

static void print_usage()
{
	printf("\nUsage: basisu filename [filename ...] <options>\n");
	
	puts("\n"
		"The default mode is compression of one or more PNG files to a .basis file. Alternate modes:\n"
		" -unpack: Use transcoder to unpack .basis file to one or more .ktx/.png files\n"
		" -validate: Validate and display information about a .basis file\n"
		" -compare: Compare two PNG images specified with -file, output PSNR and SSIM statistics and RGB/A delta images\n"
		" -version: Print basisu version and exit\n"
		"Unless an explicit mode is specified, if one or more files have the .basis extension this tool defaults to unpack mode.\n"
		"\n"
		"Important: By default, the compressor assumes the input is in the sRGB colorspace (like photos/albedo textures).\n"
		"If the input is NOT sRGB (like a normal map), be sure to specify -linear for less artifacts. Depending on the content type, some experimentation may be needed.\n"
		"\n"
		"Filenames prefixed with a @ symbol are read as filename listing files. Listing text files specify which actual filenames to process (one filename per line).\n"
		"\n"
		"Options:\n"
		" -file filename.png: Input image filename, multiple images are OK, use -file X for each input filename (prefixing input filenames with -file is optional)\n"
		" -alpha_file filename.png: Input alpha image filename, multiple images are OK, use -file X for each input filename (must be paired with -file), images converted to REC709 grayscale and used as input alpha\n"
		" -multifile_printf: printf() format strint to use to compose multiple filenames\n"
		" -multifile_first: The index of the first file to process, default is 0 (must specify -multifile_printf and -multifile_num)\n"
		" -multifile_num: The total number of files to process.\n"
		" -q X: Set quality level, 1-255, default is 128, lower=better compression/lower quality/faster, higher=less compression/higher quality/slower, default is 128. For even higher quality, use -max_endpoints/-max_selectors.\n"
		" -linear: Use linear colorspace metrics (instead of the default sRGB), and by default linear (not sRGB) mipmap filtering.\n"
		" -output_file filename: Output .basis/.ktx filename\n"
		" -output_path: Output .basis/.ktx files to specified directory.\n"
		" -debug: Enable codec debug print to stdout (slightly slower).\n"
		" -debug_images: Enable codec debug images (much slower).\n"
		" -stats: Compute and display image quality metrics (slightly slower).\n"
		" -tex_type <2d, 2darray, 3d, video, cubemap>: Set Basis file header's texture type field. Cubemap arrays require multiples of 6 images, in X+, X-, Y+, Y-, Z+, Z- order, each image must be the same resolutions.\n"
		"  2d=arbitrary 2D images, 2darray=2D array, 3D=volume texture slices, video=video frames, cubemap=array of faces. For 2darray/3d/cubemaps/video, each source image's dimensions and # of mipmap levels must be the same.\n"
		" For video, the .basis file will be written with the first frame being an I-Frame, and subsequent frames being P-Frames (using conditional replenishment). Playback must always occur in order from first to last image.\n"
		" -framerate X: Set framerate in header to X/frames sec.\n"
		" -individual: Process input images individually and output multiple .basis files (not as a texture array)\n"
		" -comp_level X: Set encoding speed vs. quality tradeoff. Range is 0-5, default is 1. Higher values=MUCH slower, but slightly higher quality. Mostly intended for videos. Use -q first!\n"
		" -fuzz_testing: Use with -validate: Disables CRC16 validation of file contents before transcoding\n"
		"\n"
		"More options:\n"
		" -max_endpoints X: Manually set the max number of color endpoint clusters from 1-16128, use instead of -q\n"
		" -max_selectors X: Manually set the max number of color selector clusters from 1-16128, use instead of -q\n"
		" -y_flip: Flip input images vertically before compression\n"
		" -normal_map: Tunes codec parameters for better quality on normal maps (linear colorspace metrics, linear mipmap filtering, no selector RDO, no sRGB)\n"
		" -no_alpha: Always output non-alpha basis files, even if one or more inputs has alpha\n"
		" -force_alpha: Always output alpha basis files, even if no inputs has alpha\n"
		" -separate_rg_to_color_alpha: Separate input R and G channels to RGB and A (for tangent space XY normal maps)\n"
		" -no_multithreading: Disable multithreading\n"
		" -no_ktx: Disable KTX writing when unpacking (faster)\n"
		" -etc1_only: Only unpack to ETC1, skipping the other texture formats during -unpack\n"
		" -disable_hierarchical_endpoint_codebooks: Disable hierarchical endpoint codebook usage, slower but higher quality on some compression levels\n"
		" -compare_ssim: Compute and display SSIM of image comparison (slow)\n"
		"\n"
		"Mipmap generation options:\n"
		" -mipmap: Generate mipmaps for each source image\n"
		" -mip_srgb: Convert image to linear before filtering, then back to sRGB\n"
		" -mip_linear: Keep image in linear light during mipmap filtering\n"
		" -mip_scale X: Set mipmap filter kernel's scale, lower=sharper, higher=more blurry, default is 1.0\n"
		" -mip_filter X: Set mipmap filter kernel, default is kaiser, filters: box, tent, bell, blackman, catmullrom, mitchell, etc.\n"
		" -mip_renorm: Renormalize normal map to unit length vectors after filtering\n"
		" -mip_clamp: Use clamp addressing on borders, instead of wrapping\n"
		" -mip_smallest X: Set smallest pixel dimension for generated mipmaps, default is 1 pixel\n"
		"By default, mipmap filtering will occur in sRGB space (for the RGB color channels) unless -linear is specified. You can override this behavior with -mip_srgb/-mip_linear.\n"
		"\n"
		"Backend endpoint/selector RDO codec options:\n"
		" -no_selector_rdo: Disable backend's selector rate distortion optimizations (slightly faster, less noisy output, but lower quality per output bit)\n"
		" -selector_rdo_thresh X: Set selector RDO quality threshold, default is 1.25, lower is higher quality but less quality per output bit (try 1.0-3.0)\n"
		" -no_endpoint_rdo: Disable backend's endpoint rate distortion optimizations (slightly faster, less noisy output, but lower quality per output bit)\n"
		" -endpoint_rdo_thresh X: Set endpoint RDO quality threshold, default is 1.5, lower is higher quality but less quality per output bit (try 1.0-3.0)\n"
		"\n"
		"Hierarchical virtual selector codebook options:\n"
		" -global_sel_pal: Always use vitual selector palettes (instead of custom palettes), slightly smaller files, but lower quality, slower encoding\n"
		" -auto_global_sel_pal: Automatically use virtual selector palettes on small images for slightly smaller files (defaults to off for faster encoding time)\n"
		" -no_hybrid_sel_cb: Don't automatically use hybrid virtual selector codebooks (for higher quality, only active when -global_sel_pal is specified)\n"
		" -global_pal_bits X: Set virtual selector codebook palette bits, range is [0,12], default is 8, higher is slower/better quality\n"
		" -global_mod_bits X: Set virtual selector codebook modifier bits, range is [0,15], defualt is 8, higher is slower/better quality\n"
		" -hybrid_sel_cb_quality_thresh X: Set hybrid selector codebook quality threshold, default is 2.0, try 1.5-3, higher is lower quality/smaller codebooks\n"
		"\n"
		"Set various fields in the Basis file header:\n"
		" -userdata0 X: Set 32-bit userdata0 field in Basis file header to X (X is a signed 32-bit int)\n"
		" -userdata1 X: Set 32-bit userdata1 field in Basis file header to X (X is a signed 32-bit int)\n"
		"\n"
		"Various command line examples:\n"
		" basisu x.png : Compress sRGB image x.png to x.basis using default settings (multiple filenames OK)\n"
		" basisu x.basis : Unpack x.basis to PNG/KTX files (multiple filenames OK)\n"
		" basisu -file x.png -mipmap -y_flip : Compress a mipmapped x.basis file from an sRGB image named x.png, Y flip each source image\n"
		" basisu -validate -file x.basis : Validate x.basis (check header, check file CRC's, attempt to transcode all slices)\n"
		" basisu -unpack -file x.basis : Validates, transcodes and unpacks x.basis to mipmapped .KTX and RGB/A .PNG files (transcodes to all supported GPU texture formats)\n"
		" basisu -q 255 -file x.png -mipmap -debug -stats : Compress sRGB x.png to x.basis at quality level 255 with compressor debug output/statistics\n"
		" basisu -linear -max_endpoints 16128 -max_selectors 16128 -file x.png : Compress non-sRGB x.png to x.basis using the largest supported manually specified codebook sizes\n"
		" basisu -linear -global_sel_pal -no_hybrid_sel_cb -file x.png : Compress a non-sRGB image, use virtual selector codebooks for improved compression (but slower encoding)\n"
		" basisu -linear -global_sel_pal -file x.png: Compress a non-sRGB image, use hybrid selector codebooks for slightly improved compression (but slower encoding)\n"
		" basisu -tex_type video -framerate 20 -multifile_printf \"x%02u.png\" -multifile_first 1 -multifile_count 20 : Compress a 20 sRGB source image video sequence (x01.png, x02.png, x03.png, etc.) to x01.basis\n"
		"\n"
		"Note: For video use, it's recommended you use a very powerful machine with many cores. Use -slower for better codebook generation, specify very large codebooks using -max_endpoints and -max_selectors, and reduce\n"
		"the default endpoint RDO threshold (-endpoint_rdo_thresh) to around 1.25. Videos may have mipmaps and alpha channels. Videos must always be played back by the transcoder in first to last image order.\n"
		"Video files currently use I-Frames on the first image, and P-Frames using conditional replenishment on subsequent frames.\n"
		"Compression level details:\n"
		" Level 0: Fastest, but has marginal quality and is a work in progress. Brittle on complex images. Avg. Y dB: 35.45\n"
		" Level 1: Hierarchical codebook searching. 36.87 dB, ~1.4x slower vs. level 0. (This is the default setting.)\n"
		" Level 2: Full codebook searching. 37.13 dB, ~1.8x slower vs. level 0. (Equivalent the the initial release's default settings.)\n"
		" Level 3: Hierarchical codebook searching, codebook k-means iterations. 37.15 dB, ~4x slower vs. level 0\n"
		" Level 4: Full codebook searching, codebook k-means iterations. 37.41 dB, ~5.5x slower vs. level 0. (Equivalent to the initial release's -slower setting.)\n"
		" Level 5: Full codebook searching, twice as many codebook k-means iterations, best ETC1 endpoint opt. 37.43 dB, ~12x slower vs. level 0\n"
	);
}

static bool load_listing_file(const std::string &f, std::vector<std::string> &filenames)
{
	std::string filename(f);
	filename.erase(0, 1);

	FILE *pFile = nullptr;
#ifdef _WIN32
	fopen_s(&pFile, filename.c_str(), "r");
#else
	pFile = fopen(filename.c_str(), "r");
#endif

	if (!pFile)
	{
		error_printf("Failed opening listing file: \"%s\"\n", filename.c_str());
		return false;
	}

	uint32_t total_filenames = 0;

	for ( ; ; )
	{
		char buf[3072];
		buf[0] = '\0';

		char *p = fgets(buf, sizeof(buf), pFile);
		if (!p)
		{
			if (ferror(pFile))
			{
				error_printf("Failed reading from listing file: \"%s\"\n", filename.c_str());

				fclose(pFile);
				return false;
			}
			else
				break;
		}

		std::string read_filename(p);
		while (read_filename.size())
		{
			if (read_filename[0] == ' ')
				read_filename.erase(0, 1);
			else 
				break;
		}

		while (read_filename.size())
		{
			const char c = read_filename.back();
			if ((c == ' ') || (c == '\n') || (c == '\r'))
				read_filename.erase(read_filename.size() - 1, 1);
			else 
				break;
		}

		if (read_filename.size())
		{
			filenames.push_back(read_filename);
			total_filenames++;
		}
	}

	fclose(pFile);

	printf("Successfully read %u filenames(s) from listing file \"%s\"\n", total_filenames, filename.c_str());

	return true;
}

class command_line_params
{
	BASISU_NO_EQUALS_OR_COPY_CONSTRUCT(command_line_params);

public:
	command_line_params() :
		m_mode(cDefault),
		m_multifile_first(0),
		m_multifile_num(0),
		m_individual(false),
		m_no_ktx(false),
		m_etc1_only(false),
		m_fuzz_testing(false),
		m_compare_ssim(false)
	{
	}

	bool parse(int arg_c, const char **arg_v)
	{
		int arg_index = 1;
		while (arg_index < arg_c)
		{
			const char *pArg = arg_v[arg_index];
			const int num_remaining_args = arg_c - (arg_index + 1);
			int arg_count = 1;

#define REMAINING_ARGS_CHECK(n) if (num_remaining_args < (n)) { error_printf("Error: Expected %u values to follow %s!\n", n, pArg); return false; }

			if (strcasecmp(pArg, "-compress") == 0)
				m_mode = cCompress;
			else if (strcasecmp(pArg, "-compare") == 0)
				m_mode = cCompare;
			else if (strcasecmp(pArg, "-unpack") == 0)
				m_mode = cUnpack;
			else if (strcasecmp(pArg, "-validate") == 0)
				m_mode = cValidate;
			else if (strcasecmp(pArg, "-version") == 0)
				m_mode = cVersion;
			else if (strcasecmp(pArg, "-compare_ssim") == 0)
				m_compare_ssim = true;
			else if (strcasecmp(pArg, "-file") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_input_filenames.push_back(std::string(arg_v[arg_index + 1]));
				arg_count++;
			}
			else if (strcasecmp(pArg, "-alpha_file") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_input_alpha_filenames.push_back(std::string(arg_v[arg_index + 1]));
				arg_count++;
			}
			else if (strcasecmp(pArg, "-multifile_printf") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_multifile_printf = std::string(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-multifile_first") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_multifile_first = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-multifile_num") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_multifile_num = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-linear") == 0)
				m_comp_params.m_perceptual = false;
			else if (strcasecmp(pArg, "-srgb") == 0)
				m_comp_params.m_perceptual = true;
			else if (strcasecmp(pArg, "-q") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_quality_level = clamp<int>(atoi(arg_v[arg_index + 1]), BASISU_QUALITY_MIN, BASISU_QUALITY_MAX);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-output_file") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_output_filename = arg_v[arg_index + 1];
				arg_count++;
			}
			else if (strcasecmp(pArg, "-output_path") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_output_path = arg_v[arg_index + 1];
				arg_count++;
			}
			else if (strcasecmp(pArg, "-debug") == 0)
			{
				m_comp_params.m_debug = true;
				enable_debug_printf(true);
			}
			else if (strcasecmp(pArg, "-debug_images") == 0)
				m_comp_params.m_debug_images = true;
			else if (strcasecmp(pArg, "-stats") == 0)
				m_comp_params.m_compute_stats = true;
			else if (strcasecmp(pArg, "-comp_level") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_compression_level = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-slower") == 0)
			{
				// This option is gone, but we'll do something reasonable with it anyway. Level 4 is equivalent to the original release's -slower, but let's just go to level 2.
				m_comp_params.m_compression_level = 2;
			}
			else if (strcasecmp(pArg, "-max_endpoints") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_max_endpoint_clusters = clamp<int>(atoi(arg_v[arg_index + 1]), 1, BASISU_MAX_ENDPOINT_CLUSTERS);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-max_selectors") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_max_selector_clusters = clamp<int>(atoi(arg_v[arg_index + 1]), 1, BASISU_MAX_SELECTOR_CLUSTERS);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-y_flip") == 0)
				m_comp_params.m_y_flip = true;
			else if (strcasecmp(pArg, "-normal_map") == 0)
			{
				m_comp_params.m_perceptual = false;
				m_comp_params.m_mip_srgb = false;
				m_comp_params.m_no_selector_rdo = true;
				m_comp_params.m_no_endpoint_rdo = true;
			}
			else if (strcasecmp(pArg, "-no_alpha") == 0)
				m_comp_params.m_check_for_alpha = false;
			else if (strcasecmp(pArg, "-force_alpha") == 0)
				m_comp_params.m_force_alpha = true;
			else if ((strcasecmp(pArg, "-separate_rg_to_color_alpha") == 0) ||
			        (strcasecmp(pArg, "-seperate_rg_to_color_alpha") == 0)) // was mispelled for a while - whoops!
				m_comp_params.m_seperate_rg_to_color_alpha = true;
			else if (strcasecmp(pArg, "-no_multithreading") == 0)
			{
				m_comp_params.m_multithreading = false;
			}
			else if (strcasecmp(pArg, "-mipmap") == 0)
				m_comp_params.m_mip_gen = true;
			else if (strcasecmp(pArg, "-no_ktx") == 0)
				m_no_ktx = true;
			else if (strcasecmp(pArg, "-etc1_only") == 0)
				m_etc1_only = true;
			else if (strcasecmp(pArg, "-disable_hierarchical_endpoint_codebooks") == 0)
				m_comp_params.m_disable_hierarchical_endpoint_codebooks = true;
			else if (strcasecmp(pArg, "-mip_scale") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_mip_scale = (float)atof(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-mip_filter") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_mip_filter = arg_v[arg_index + 1];
				// TODO: Check filter 
				arg_count++;
			}
			else if (strcasecmp(pArg, "-mip_renorm") == 0)
				m_comp_params.m_mip_renormalize = true;
			else if (strcasecmp(pArg, "-mip_clamp") == 0)
				m_comp_params.m_mip_wrapping = false;
			else if (strcasecmp(pArg, "-mip_smallest") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_mip_smallest_dimension = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-mip_srgb") == 0)
				m_comp_params.m_mip_srgb = true;
			else if (strcasecmp(pArg, "-mip_linear") == 0)
				m_comp_params.m_mip_srgb = false;
			else if (strcasecmp(pArg, "-no_selector_rdo") == 0)
				m_comp_params.m_no_selector_rdo = true;
			else if (strcasecmp(pArg, "-selector_rdo_thresh") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_selector_rdo_thresh = (float)atof(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-no_endpoint_rdo") == 0)
				m_comp_params.m_no_endpoint_rdo = true;
			else if (strcasecmp(pArg, "-endpoint_rdo_thresh") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_endpoint_rdo_thresh = (float)atof(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-global_sel_pal") == 0)
				m_comp_params.m_global_sel_pal = true;
			else if (strcasecmp(pArg, "-no_auto_global_sel_pal") == 0)
				m_comp_params.m_auto_global_sel_pal = false;
			else if (strcasecmp(pArg, "-auto_global_sel_pal") == 0)
				m_comp_params.m_auto_global_sel_pal = true;
			else if (strcasecmp(pArg, "-global_pal_bits") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_global_pal_bits = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-global_mod_bits") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_global_mod_bits = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-no_hybrid_sel_cb") == 0)
				m_comp_params.m_no_hybrid_sel_cb = true;
			else if (strcasecmp(pArg, "-hybrid_sel_cb_quality_thresh") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_hybrid_sel_cb_quality_thresh = (float)atof(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-userdata0") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_userdata0 = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-userdata1") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_comp_params.m_userdata1 = atoi(arg_v[arg_index + 1]);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-framerate") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				double fps = atof(arg_v[arg_index + 1]);
				double us_per_frame = 0;
				if (fps > 0)
					us_per_frame = 1000000.0f / fps;

				m_comp_params.m_us_per_frame = clamp<int>(static_cast<int>(us_per_frame + .5f), 0, basist::cBASISMaxUSPerFrame);
				arg_count++;
			}
			else if (strcasecmp(pArg, "-tex_type") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				const char *pType = arg_v[arg_index + 1];
				if (strcasecmp(pType, "2d") == 0)
					m_comp_params.m_tex_type = basist::cBASISTexType2D;
				else if (strcasecmp(pType, "2darray") == 0)
					m_comp_params.m_tex_type = basist::cBASISTexType2DArray;
				else if (strcasecmp(pType, "3d") == 0)
					m_comp_params.m_tex_type = basist::cBASISTexTypeVolume;
				else if (strcasecmp(pType, "cubemap") == 0)
					m_comp_params.m_tex_type = basist::cBASISTexTypeCubemapArray;
				else if (strcasecmp(pType, "video") == 0)
					m_comp_params.m_tex_type = basist::cBASISTexTypeVideoFrames;
				else
				{
					error_printf("Invalid texture type: %s\n", pType);
					return false;
				}
				arg_count++;
			}
			else if (strcasecmp(pArg, "-individual") == 0)
				m_individual = true;
			else if (strcasecmp(pArg, "-fuzz_testing") == 0)
				m_fuzz_testing = true;
			else if (strcasecmp(pArg, "-csv_file") == 0)
			{
				REMAINING_ARGS_CHECK(1);
				m_csv_file = arg_v[arg_index + 1];
				m_comp_params.m_compute_stats = true;

				arg_count++;
			}
			else if (pArg[0] == '-')
			{
				error_printf("Unrecognized command line option: %s\n", pArg);
				return false;
			}
			else
			{
				// Let's assume it's a source filename, so globbing works
				//error_printf("Unrecognized command line option: %s\n", pArg);
				m_input_filenames.push_back(pArg);
			}

			arg_index += arg_count;
		}
		
		if (m_comp_params.m_quality_level != -1)
		{
			m_comp_params.m_max_endpoint_clusters = 0;
			m_comp_params.m_max_selector_clusters = 0;
		}
		else if ((!m_comp_params.m_max_endpoint_clusters) || (!m_comp_params.m_max_selector_clusters))
		{
			m_comp_params.m_max_endpoint_clusters = 0;
			m_comp_params.m_max_selector_clusters = 0;

			m_comp_params.m_quality_level = 128;
		}

		if (!m_comp_params.m_mip_srgb.was_changed())
		{
			// They didn't specify what colorspace to do mipmap filtering in, so choose sRGB if they've specified that the texture is sRGB.
			if (m_comp_params.m_perceptual)
				m_comp_params.m_mip_srgb = true;
			else
				m_comp_params.m_mip_srgb = false;
		}
				
		return true;
	}

	bool process_listing_files()
	{
		std::vector<std::string> new_input_filenames;
		for (uint32_t i = 0; i < m_input_filenames.size(); i++)
		{
			if (m_input_filenames[i][0] == '@')
			{
				if (!load_listing_file(m_input_filenames[i], new_input_filenames))
					return false;
			}
			else
				new_input_filenames.push_back(m_input_filenames[i]);
		}
		new_input_filenames.swap(m_input_filenames);

		std::vector<std::string> new_input_alpha_filenames;
		for (uint32_t i = 0; i < m_input_alpha_filenames.size(); i++)
		{
			if (m_input_alpha_filenames[i][0] == '@')
			{
				if (!load_listing_file(m_input_alpha_filenames[i], new_input_alpha_filenames))
					return false;
			}
			else
				new_input_alpha_filenames.push_back(m_input_alpha_filenames[i]);
		}
		new_input_alpha_filenames.swap(m_input_alpha_filenames);
		
		return true;
	}

	basis_compressor_params m_comp_params;
		
	tool_mode m_mode;
		
	std::vector<std::string> m_input_filenames;
	std::vector<std::string> m_input_alpha_filenames;

	std::string m_output_filename;
	std::string m_output_path;

	std::string m_multifile_printf;
	uint32_t m_multifile_first;
	uint32_t m_multifile_num;

	std::string m_csv_file;

	bool m_individual;
	bool m_no_ktx;
	bool m_etc1_only;
	bool m_fuzz_testing;
	bool m_compare_ssim;
};

static bool expand_multifile(command_line_params &opts)
{
	if (!opts.m_multifile_printf.size())
		return true;
	
	if (!opts.m_multifile_num)
	{
		error_printf("-multifile_printf specified, but not -multifile_num\n");
		return false;
	}
	
	std::string fmt(opts.m_multifile_printf);
	size_t x = fmt.find_first_of('!');
	if (x != std::string::npos)
		fmt[x] = '%';

	if (string_find_right(fmt, '%') == -1)
	{
		error_printf("Must include C-style printf() format character '%%' in -multifile_printf string\n");
		return false;
	}
		
	for (uint32_t i = opts.m_multifile_first; i < opts.m_multifile_first + opts.m_multifile_num; i++)
	{
		char buf[1024];
#ifdef _WIN32		
		sprintf_s(buf, sizeof(buf), fmt.c_str(), i);
#else
		snprintf(buf, sizeof(buf), fmt.c_str(), i);
#endif		

		if (buf[0])
			opts.m_input_filenames.push_back(buf);
	}

	return true;
}

static bool compress_mode(command_line_params &opts)
{
	basist::etc1_global_selector_codebook sel_codebook(basist::g_global_selector_cb_size, basist::g_global_selector_cb);

	uint32_t num_threads = 1;

	if (opts.m_comp_params.m_multithreading)
	{
		num_threads = std::thread::hardware_concurrency();
		if (num_threads < 1)
			num_threads = 1;
	}

	job_pool jpool(num_threads);
	opts.m_comp_params.m_pJob_pool = &jpool;
		
	if (!expand_multifile(opts))
	{
		error_printf("-multifile expansion failed!\n");
		return false;
	}

	if (!opts.m_input_filenames.size())
	{
		error_printf("No input files to process!\n");
		return false;
	}
						
	basis_compressor_params &params = opts.m_comp_params;

	params.m_read_source_images = true;
	params.m_write_output_basis_files = true;
	params.m_pSel_codebook = &sel_codebook;

	FILE *pCSV_file = nullptr;
	if (opts.m_csv_file.size())
	{
		pCSV_file = fopen_safe(opts.m_csv_file.c_str(), "a");
		if (!pCSV_file)
		{
			error_printf("Failed opening CVS file \"%s\"\n", opts.m_csv_file.c_str());
			return false;
		}
	}

	printf("Processing %u total files\n", (uint32_t)opts.m_input_filenames.size());
				
	for (size_t file_index = 0; file_index < (opts.m_individual ? opts.m_input_filenames.size() : 1U); file_index++)
	{
		if (opts.m_individual)
		{
			params.m_source_filenames.resize(1);
			params.m_source_filenames[0] = opts.m_input_filenames[file_index];

			if (file_index < opts.m_input_alpha_filenames.size()) 
			{
				params.m_source_alpha_filenames.resize(1);
				params.m_source_alpha_filenames[0] = opts.m_input_alpha_filenames[file_index];
				
				printf("Processing source file \"%s\", alpha file \"%s\"\n", params.m_source_filenames[0].c_str(), params.m_source_alpha_filenames[0].c_str());
			}
			else
			{
				params.m_source_alpha_filenames.resize(0);
				
				printf("Processing source file \"%s\"\n", params.m_source_filenames[0].c_str());
			}
		}
		else
		{
			params.m_source_filenames = opts.m_input_filenames;
			params.m_source_alpha_filenames = opts.m_input_alpha_filenames;
		}
				
		if ((opts.m_output_filename.size()) && (!opts.m_individual))
			params.m_out_filename = opts.m_output_filename;
		else 
		{
			std::string filename;
		
			string_get_filename(opts.m_input_filenames[file_index].c_str(), filename);
			string_remove_extension(filename);
			filename += ".basis";

			if (opts.m_output_path.size())
				string_combine_path(filename, opts.m_output_path.c_str(), filename.c_str());
		
			params.m_out_filename = filename;
		}
		
		basis_compressor c;

		if (!c.init(opts.m_comp_params))
		{
			error_printf("basis_compressor::init() failed!\n");

			if (pCSV_file)
			{
				fclose(pCSV_file);
				pCSV_file = nullptr;
			}

			return false;
		}

		interval_timer tm;
		tm.start();

		basis_compressor::error_code ec = c.process();

		tm.stop();

		if (ec == basis_compressor::cECSuccess)
		{
			printf("Compression succeeded to file \"%s\" in %3.3f secs\n", params.m_out_filename.c_str(), tm.get_elapsed_secs());
		}
		else
		{
			bool exit_flag = true;

			switch (ec)
			{
				case basis_compressor::cECFailedReadingSourceImages:
				{
					error_printf("Compressor failed reading a source image!\n");
					
					if (opts.m_individual)
						exit_flag = false;

					break;
				}
				case basis_compressor::cECFailedValidating:
					error_printf("Compressor failed 2darray/cubemap/video validation checks!\n");
					break;
				case basis_compressor::cECFailedFrontEnd:
					error_printf("Compressor frontend stage failed!\n");
					break;
				case basis_compressor::cECFailedFontendExtract:
					error_printf("Compressor frontend data extraction failed!\n");
					break;
				case basis_compressor::cECFailedBackend:
					error_printf("Compressor backend stage failed!\n");
					break;
				case basis_compressor::cECFailedCreateBasisFile:
					error_printf("Compressor failed creating Basis file data!\n");
					break;
				case basis_compressor::cECFailedWritingOutput:
					error_printf("Compressor failed writing to output Basis file!\n");
					break;
				default:
					error_printf("basis_compress::process() failed!\n");
					break;
			}
		
			if (exit_flag)
			{
				if (pCSV_file)
				{
					fclose(pCSV_file);
					pCSV_file = nullptr;
				}

				return false;
			}
		}

		if ((pCSV_file) && (c.get_stats().size()))
		{
			for (size_t slice_index = 0; slice_index < c.get_stats().size(); slice_index++)
			{
				fprintf(pCSV_file, "\"%s\", %u, %u, %u, %u, %u, %f, %f, %f, %f, %u, %u, %f\n",
					params.m_out_filename.c_str(),
					(uint32_t)slice_index, (uint32_t)c.get_stats().size(),
					c.get_stats()[slice_index].m_width, c.get_stats()[slice_index].m_height, (uint32_t)c.get_any_source_image_has_alpha(),
					c.get_basis_bits_per_texel(),
					c.get_stats()[slice_index].m_best_luma_709_psnr,
					c.get_stats()[slice_index].m_basis_etc1s_luma_709_psnr,
					c.get_stats()[slice_index].m_basis_bc1_luma_709_psnr,
					params.m_quality_level, (int)params.m_compression_level, tm.get_elapsed_secs());
				fflush(pCSV_file);
			}
		}
				
		if (opts.m_individual)
			printf("\n");

	} // file_index

	if (pCSV_file)
	{
		fclose(pCSV_file);
		pCSV_file = nullptr;
	}
		
	return true;
}

static bool unpack_and_validate_mode(command_line_params &opts, bool validate_flag)
{
	basist::etc1_global_selector_codebook sel_codebook(basist::g_global_selector_cb_size, basist::g_global_selector_cb);

	if (!opts.m_input_filenames.size())
	{
		error_printf("No input files to process!\n");
		return false;
	}

	uint32_t total_unpack_warnings = 0;
	uint32_t total_pvrtc_nonpow2_warnings = 0;

	for (uint32_t file_index = 0; file_index < opts.m_input_filenames.size(); file_index++)
	{
		const char* pInput_filename = opts.m_input_filenames[file_index].c_str();

		std::string base_filename;
		string_split_path(pInput_filename, nullptr, nullptr, &base_filename, nullptr);

		uint8_vec basis_data;
		if (!basisu::read_file_to_vec(pInput_filename, basis_data))
		{
			error_printf("Failed reading file \"%s\"\n", pInput_filename);
			return false;
		}

		printf("Input file \"%s\"\n", pInput_filename);

		if (!basis_data.size())
		{
			error_printf("File is empty!\n");
			return false;
		}

		if (basis_data.size() > UINT32_MAX)
		{
			error_printf("File is too large!\n");
			return false;
		}

		basist::basisu_transcoder dec(&sel_codebook);

		if (!opts.m_fuzz_testing)
		{
			// Skip the full validation, which CRC16's the entire file.

			// Validate the file - note this isn't necessary for transcoding
			if (!dec.validate_file_checksums(&basis_data[0], (uint32_t)basis_data.size(), true))
			{
				error_printf("File version is unsupported, or file fail CRC checks!\n");
				return false;
			}
		}

		printf("File version and CRC checks succeeded\n");

		basist::basisu_file_info fileinfo;
		if (!dec.get_file_info(&basis_data[0], (uint32_t)basis_data.size(), fileinfo))
		{
			error_printf("Failed retrieving Basis file information!\n");
			return false;
		}

		assert(fileinfo.m_total_images == fileinfo.m_image_mipmap_levels.size());
		assert(fileinfo.m_total_images == dec.get_total_images(&basis_data[0], (uint32_t)basis_data.size()));

		printf("File info:\n");
		printf("  Version: %X\n", fileinfo.m_version);
		printf("  Total header size: %u\n", fileinfo.m_total_header_size);
		printf("  Total selectors: %u\n", fileinfo.m_total_selectors);
		printf("  Selector codebook size: %u\n", fileinfo.m_selector_codebook_size);
		printf("  Total endpoints: %u\n", fileinfo.m_total_endpoints);
		printf("  Endpoint codebook size: %u\n", fileinfo.m_endpoint_codebook_size);
		printf("  Tables size: %u\n", fileinfo.m_tables_size);
		printf("  Slices size: %u\n", fileinfo.m_slices_size);
		printf("  Texture type: %s\n", basist::basis_get_texture_type_name(fileinfo.m_tex_type));
		printf("  us per frame: %u (%f fps)\n", fileinfo.m_us_per_frame, fileinfo.m_us_per_frame ? (1.0f / ((float)fileinfo.m_us_per_frame / 1000000.0f)) : 0.0f);
		printf("  Total slices: %u\n", (uint32_t)fileinfo.m_slice_info.size());
		printf("  Total images: %i\n", fileinfo.m_total_images);
		printf("  Y Flipped: %u, Has alpha slices: %u\n", fileinfo.m_y_flipped, fileinfo.m_has_alpha_slices);
		printf("  userdata0: 0x%X userdata1: 0x%X\n", fileinfo.m_userdata0, fileinfo.m_userdata1);
		printf("  Per-image mipmap levels: ");
		for (uint32_t i = 0; i < fileinfo.m_total_images; i++)
			printf("%u ", fileinfo.m_image_mipmap_levels[i]);
		printf("\n");

		printf("\nImage info:\n");
		for (uint32_t i = 0; i < fileinfo.m_total_images; i++)
		{
			basist::basisu_image_info ii;
			if (!dec.get_image_info(&basis_data[0], (uint32_t)basis_data.size(), ii, i))
			{
				error_printf("get_image_info() failed!\n");
				return false;
			}

			printf("Image %u: MipLevels: %u OrigDim: %ux%u, BlockDim: %ux%u, FirstSlice: %u, HasAlpha: %u\n", i, ii.m_total_levels, ii.m_orig_width, ii.m_orig_height,
				ii.m_num_blocks_x, ii.m_num_blocks_y, ii.m_first_slice_index, (uint32_t)ii.m_alpha_flag);
		}

		printf("\nSlice info:\n");
		for (uint32_t i = 0; i < fileinfo.m_slice_info.size(); i++)
		{
			const basist::basisu_slice_info& sliceinfo = fileinfo.m_slice_info[i];
			printf("%u: OrigWidthHeight: %ux%u, BlockDim: %ux%u, TotalBlocks: %u, Compressed size: %u, Image: %u, Level: %u, UnpackedCRC16: 0x%X, alpha: %u, iframe: %i\n",
				i,
				sliceinfo.m_orig_width, sliceinfo.m_orig_height,
				sliceinfo.m_num_blocks_x, sliceinfo.m_num_blocks_y,
				sliceinfo.m_total_blocks,
				sliceinfo.m_compressed_size,
				sliceinfo.m_image_index, sliceinfo.m_level_index,
				sliceinfo.m_unpacked_slice_crc16,
				(uint32_t)sliceinfo.m_alpha_flag,
				(uint32_t)sliceinfo.m_iframe_flag);
		}
		printf("\n");

		interval_timer tm;
		tm.start();

		if (!dec.start_transcoding(&basis_data[0], (uint32_t)basis_data.size()))
		{
			error_printf("start_transcoding() failed!\n");
			return false;
		}

		printf("start_transcoding time: %3.3f ms\n", tm.get_elapsed_ms());
				
		std::vector< gpu_image_vec > gpu_images[(int)basist::transcoder_texture_format::cTFTotalTextureFormats];
		
		int first_format = 0;
		int last_format = (int)basist::transcoder_texture_format::cTFTotalTextureFormats;

		if (opts.m_etc1_only)
		{
			first_format = (int)basist::transcoder_texture_format::cTFETC1_RGB;
			last_format = first_format + 1;
		}

		for (int format_iter = first_format; format_iter < last_format; format_iter++)
		{
			basist::transcoder_texture_format tex_fmt = static_cast<basist::transcoder_texture_format>(format_iter);

			if (basist::basis_transcoder_format_is_uncompressed(tex_fmt))
				continue;
			
			gpu_images[(int)tex_fmt].resize(fileinfo.m_total_images);

			for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index++)
				gpu_images[(int)tex_fmt][image_index].resize(fileinfo.m_image_mipmap_levels[image_index]);
		}

		// Now transcode the file to all supported texture formats and save mipmapped KTX files
		for (int format_iter = first_format; format_iter < last_format; format_iter++)
		{
			const basist::transcoder_texture_format transcoder_tex_fmt = static_cast<basist::transcoder_texture_format>(format_iter);

			if (basist::basis_transcoder_format_is_uncompressed(transcoder_tex_fmt))
				continue;

			for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index++)
			{
				for (uint32_t level_index = 0; level_index < fileinfo.m_image_mipmap_levels[image_index]; level_index++)
				{
					basist::basisu_image_level_info level_info;

					if (!dec.get_image_level_info(&basis_data[0], (uint32_t)basis_data.size(), level_info, image_index, level_index))
					{
						error_printf("Failed retrieving image level information (%u %u)!\n", image_index, level_index);
						return false;
					}
										
					if ((transcoder_tex_fmt == basist::transcoder_texture_format::cTFPVRTC1_4_RGB) || (transcoder_tex_fmt == basist::transcoder_texture_format::cTFPVRTC1_4_RGBA))
					{
						if (!is_pow2(level_info.m_width) || !is_pow2(level_info.m_height))
						{
							total_pvrtc_nonpow2_warnings++;

							printf("Warning: Will not transcode image %u level %u res %ux%u to PVRTC1 (one or more dimension is not a power of 2)\n", image_index, level_index, level_info.m_width, level_info.m_height);

							// Can't transcode this image level to PVRTC because it's not a pow2 (we're going to support transcoding non-pow2 to the next larger pow2 soon)
							continue;
						}
					}

					basisu::texture_format tex_fmt = basis_get_basisu_texture_format(transcoder_tex_fmt);

					gpu_image& gi = gpu_images[(int)transcoder_tex_fmt][image_index][level_index];
					gi.init(tex_fmt, level_info.m_orig_width, level_info.m_orig_height);

					// Fill the buffer with psuedo-random bytes, to help more visibly detect cases where the transcoder fails to write to part of the output.
					fill_buffer_with_random_bytes(gi.get_ptr(), gi.get_size_in_bytes());

					uint32_t decode_flags = 0;

					tm.start();
														
					if (!dec.transcode_image_level(&basis_data[0], (uint32_t)basis_data.size(), image_index, level_index, gi.get_ptr(), gi.get_total_blocks(), transcoder_tex_fmt, decode_flags))
					{
						error_printf("Failed transcoding image level (%u %u %u)!\n", image_index, level_index, format_iter);
						return false;
					}
					
					double total_transcode_time = tm.get_elapsed_ms();

					printf("Transcode of image %u level %u res %ux%u format %s succeeded in %3.3f ms\n", image_index, level_index, level_info.m_orig_width, level_info.m_orig_height, basist::basis_get_format_name(transcoder_tex_fmt), total_transcode_time);

				} // format_iter

			} // level_index

		} // image_info

		if (!validate_flag)
		{
			// Now write KTX files and unpack them to individual PNG's

			for (int format_iter = first_format; format_iter < last_format; format_iter++)
			{
				const basist::transcoder_texture_format transcoder_tex_fmt = static_cast<basist::transcoder_texture_format>(format_iter);

				if (basist::basis_transcoder_format_is_uncompressed(transcoder_tex_fmt))
					continue;

				if ((!opts.m_no_ktx) && (fileinfo.m_tex_type == basist::cBASISTexTypeCubemapArray))
				{
					// No KTX tool that we know of supports cubemap arrays, so write individual cubemap files.
					for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index += 6)
					{
						std::vector<gpu_image_vec> cubemap;
						for (uint32_t i = 0; i < 6; i++)
							cubemap.push_back(gpu_images[format_iter][image_index + i]);

						std::string ktx_filename(base_filename + string_format("_transcoded_cubemap_%s_%u.ktx", basist::basis_get_format_name(transcoder_tex_fmt), image_index / 6));
						if (!write_compressed_texture_file(ktx_filename.c_str(), cubemap, true))
						{
							error_printf("Failed writing KTX file \"%s\"!\n", ktx_filename.c_str());
							return false;
						}
						printf("Wrote KTX file \"%s\"\n", ktx_filename.c_str());
					}
				}

				for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index++)
				{
					gpu_image_vec& gi = gpu_images[format_iter][image_index];

					if (!gi.size())
						continue;

					uint32_t level;
					for (level = 0; level < gi.size(); level++)
						if (!gi[level].get_total_blocks())
							break;

					if (level < gi.size())
						continue;

					if ((!opts.m_no_ktx) && (fileinfo.m_tex_type != basist::cBASISTexTypeCubemapArray))
					{
						std::string ktx_filename(base_filename + string_format("_transcoded_%s_%04u.ktx", basist::basis_get_format_name(transcoder_tex_fmt), image_index));
						if (!write_compressed_texture_file(ktx_filename.c_str(), gi))
						{
							error_printf("Failed writing KTX file \"%s\"!\n", ktx_filename.c_str());
							return false;
						}
						printf("Wrote KTX file \"%s\"\n", ktx_filename.c_str());
					}

					for (uint32_t level_index = 0; level_index < gi.size(); level_index++)
					{
						basist::basisu_image_level_info level_info;

						if (!dec.get_image_level_info(&basis_data[0], (uint32_t)basis_data.size(), level_info, image_index, level_index))
						{
							error_printf("Failed retrieving image level information (%u %u)!\n", image_index, level_index);
							return false;
						}

						image u;
						if (!gi[level_index].unpack(u))
						{
							printf("Warning: Failed unpacking GPU texture data (%u %u %u). Unpacking as much as possible.\n", format_iter, image_index, level_index);
							total_unpack_warnings++;
						}
						//u.crop(level_info.m_orig_width, level_info.m_orig_height);

						std::string rgb_filename;
						if (gi.size() > 1)
							rgb_filename = base_filename + string_format("_unpacked_rgb_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index);
						else
							rgb_filename = base_filename + string_format("_unpacked_rgb_%s_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), image_index);
						if (!save_png(rgb_filename, u, cImageSaveIgnoreAlpha))
						{
							error_printf("Failed writing to PNG file \"%s\"\n", rgb_filename.c_str());
							return false;
						}
						printf("Wrote PNG file \"%s\"\n", rgb_filename.c_str());

						if (transcoder_tex_fmt == basist::transcoder_texture_format::cTFFXT1_RGB)
						{
							std::string out_filename;
							if (gi.size() > 1)
								out_filename = base_filename + string_format("_unpacked_rgb_%s_%u_%04u.out", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index);
							else
								out_filename = base_filename + string_format("_unpacked_rgb_%s_%04u.out", basist::basis_get_format_name(transcoder_tex_fmt), image_index);
							if (!write_3dfx_out_file(out_filename.c_str(), gi[level_index]))
							{
								error_printf("Failed writing to OUT file \"%s\"\n", out_filename.c_str());
								return false;
							}
							printf("Wrote .OUT file \"%s\"\n", out_filename.c_str());
						}

						if (basis_transcoder_format_has_alpha(transcoder_tex_fmt))
						{
							std::string a_filename;
							if (gi.size() > 1)
								a_filename = base_filename + string_format("_unpacked_a_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index);
							else
								a_filename = base_filename + string_format("_unpacked_a_%s_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), image_index);
							if (!save_png(a_filename, u, cImageSaveGrayscale, 3))
							{
								error_printf("Failed writing to PNG file \"%s\"\n", a_filename.c_str());
								return false;
							}
							printf("Wrote PNG file \"%s\"\n", a_filename.c_str());
						}

					} // level_index

				} // image_index

			} // format_iter

		} // if (!validate_flag)

		// Now unpack to RGBA using the transcoder itself to do the unpacking to raster images
		for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index++)
		{
			for (uint32_t level_index = 0; level_index < fileinfo.m_image_mipmap_levels[image_index]; level_index++)
			{
				const basist::transcoder_texture_format transcoder_tex_fmt = basist::transcoder_texture_format::cTFRGBA32;

				basist::basisu_image_level_info level_info;

				if (!dec.get_image_level_info(&basis_data[0], (uint32_t)basis_data.size(), level_info, image_index, level_index))
				{
					error_printf("Failed retrieving image level information (%u %u)!\n", image_index, level_index);
					return false;
				}

				image img(level_info.m_orig_width, level_info.m_orig_height);

				fill_buffer_with_random_bytes(&img(0, 0), img.get_total_pixels() * sizeof(uint32_t));

				tm.start();

				if (!dec.transcode_image_level(&basis_data[0], (uint32_t)basis_data.size(), image_index, level_index, &img(0, 0).r, img.get_total_pixels(), transcoder_tex_fmt, 0, img.get_pitch(), nullptr, img.get_height()))
				{
					error_printf("Failed transcoding image level (%u %u %u)!\n", image_index, level_index, transcoder_tex_fmt);
					return false;
				}

				double total_transcode_time = tm.get_elapsed_ms();
								
				printf("Transcode of image %u level %u res %ux%u format %s succeeded in %3.3f ms\n", image_index, level_index, level_info.m_orig_width, level_info.m_orig_height, basist::basis_get_format_name(transcoder_tex_fmt), total_transcode_time);

				std::string rgb_filename(base_filename + string_format("_unpacked_rgb_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index));
				if (!save_png(rgb_filename, img, cImageSaveIgnoreAlpha))
				{
					error_printf("Failed writing to PNG file \"%s\"\n", rgb_filename.c_str());
					return false;
				}
				printf("Wrote PNG file \"%s\"\n", rgb_filename.c_str());

				std::string a_filename(base_filename + string_format("_unpacked_a_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index));
				if (!save_png(a_filename, img, cImageSaveGrayscale, 3))
				{
					error_printf("Failed writing to PNG file \"%s\"\n", a_filename.c_str());
					return false;
				}
				printf("Wrote PNG file \"%s\"\n", a_filename.c_str());

			} // level_index
		} // image_index

		// Now unpack to RGB565 using the transcoder itself to do the unpacking to raster images
		for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index++)
		{
			for (uint32_t level_index = 0; level_index < fileinfo.m_image_mipmap_levels[image_index]; level_index++)
			{
				const basist::transcoder_texture_format transcoder_tex_fmt = basist::transcoder_texture_format::cTFRGB565;

				basist::basisu_image_level_info level_info;

				if (!dec.get_image_level_info(&basis_data[0], (uint32_t)basis_data.size(), level_info, image_index, level_index))
				{
					error_printf("Failed retrieving image level information (%u %u)!\n", image_index, level_index);
					return false;
				}

				std::vector<uint16_t> packed_img(level_info.m_orig_width * level_info.m_orig_height);

				fill_buffer_with_random_bytes(&packed_img[0], packed_img.size() * sizeof(uint16_t));

				tm.start();

				if (!dec.transcode_image_level(&basis_data[0], (uint32_t)basis_data.size(), image_index, level_index, &packed_img[0], (uint32_t)packed_img.size(), transcoder_tex_fmt, 0, level_info.m_orig_width, nullptr, level_info.m_orig_height))
				{
					error_printf("Failed transcoding image level (%u %u %u)!\n", image_index, level_index, transcoder_tex_fmt);
					return false;
				}

				double total_transcode_time = tm.get_elapsed_ms();

				image img(level_info.m_orig_width, level_info.m_orig_height);
				for (uint32_t y = 0; y < level_info.m_orig_height; y++)
				{
					for (uint32_t x = 0; x < level_info.m_orig_width; x++)
					{
						const uint16_t p = packed_img[x + y * level_info.m_orig_width];
						uint32_t r = p >> 11, g = (p >> 5) & 63, b = p & 31;
						r = (r << 3) | (r >> 2);
						g = (g << 2) | (g >> 4);
						b = (b << 3) | (b >> 2);
						img(x, y).set(r, g, b, 255);
					}
				}

				printf("Transcode of image %u level %u res %ux%u format %s succeeded in %3.3f ms\n", image_index, level_index, level_info.m_orig_width, level_info.m_orig_height, basist::basis_get_format_name(transcoder_tex_fmt), total_transcode_time);

				std::string rgb_filename(base_filename + string_format("_unpacked_rgb_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index));
				if (!save_png(rgb_filename, img, cImageSaveIgnoreAlpha))
				{
					error_printf("Failed writing to PNG file \"%s\"\n", rgb_filename.c_str());
					return false;
				}
				printf("Wrote PNG file \"%s\"\n", rgb_filename.c_str());

			} // level_index
		} // image_index

		// Now unpack to RGBA4444 using the transcoder itself to do the unpacking to raster images
		for (uint32_t image_index = 0; image_index < fileinfo.m_total_images; image_index++)
		{
			for (uint32_t level_index = 0; level_index < fileinfo.m_image_mipmap_levels[image_index]; level_index++)
			{
				const basist::transcoder_texture_format transcoder_tex_fmt = basist::transcoder_texture_format::cTFRGBA4444;

				basist::basisu_image_level_info level_info;

				if (!dec.get_image_level_info(&basis_data[0], (uint32_t)basis_data.size(), level_info, image_index, level_index))
				{
					error_printf("Failed retrieving image level information (%u %u)!\n", image_index, level_index);
					return false;
				}

				std::vector<uint16_t> packed_img(level_info.m_orig_width * level_info.m_orig_height);

				fill_buffer_with_random_bytes(&packed_img[0], packed_img.size() * sizeof(uint16_t));

				tm.start();

				if (!dec.transcode_image_level(&basis_data[0], (uint32_t)basis_data.size(), image_index, level_index, &packed_img[0], (uint32_t)packed_img.size(), transcoder_tex_fmt, 0, level_info.m_orig_width, nullptr, level_info.m_orig_height))
				{
					error_printf("Failed transcoding image level (%u %u %u)!\n", image_index, level_index, transcoder_tex_fmt);
					return false;
				}

				double total_transcode_time = tm.get_elapsed_ms();

				image img(level_info.m_orig_width, level_info.m_orig_height);
				for (uint32_t y = 0; y < level_info.m_orig_height; y++)
				{
					for (uint32_t x = 0; x < level_info.m_orig_width; x++)
					{
						const uint16_t p = packed_img[x + y * level_info.m_orig_width];
						uint32_t r = p >> 12, g = (p >> 8) & 15, b = (p >> 4) & 15, a = p & 15;
						r = (r << 4) | r;
						g = (g << 4) | g;
						b = (b << 4) | b;
						a = (a << 4) | a;
						img(x, y).set(r, g, b, a);
					}
				}

				printf("Transcode of image %u level %u res %ux%u format %s succeeded in %3.3f ms\n", image_index, level_index, level_info.m_orig_width, level_info.m_orig_height, basist::basis_get_format_name(transcoder_tex_fmt), total_transcode_time);

				std::string rgb_filename(base_filename + string_format("_unpacked_rgb_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index));
				if (!save_png(rgb_filename, img, cImageSaveIgnoreAlpha))
				{
					error_printf("Failed writing to PNG file \"%s\"\n", rgb_filename.c_str());
					return false;
				}
				printf("Wrote PNG file \"%s\"\n", rgb_filename.c_str());

				std::string a_filename(base_filename + string_format("_unpacked_a_%s_%u_%04u.png", basist::basis_get_format_name(transcoder_tex_fmt), level_index, image_index));
				if (!save_png(a_filename, img, cImageSaveGrayscale, 3))
				{
					error_printf("Failed writing to PNG file \"%s\"\n", a_filename.c_str());
					return false;
				}
				printf("Wrote PNG file \"%s\"\n", a_filename.c_str());

			} // level_index
		} // image_index

	} // file_index

	if (total_pvrtc_nonpow2_warnings)
		printf("Warning: %u images could not be transcoded to PVRTC1 because one or both dimensions were not a power of 2\n", total_pvrtc_nonpow2_warnings);

	if (total_unpack_warnings)
		printf("ATTENTION: %u total images had invalid GPU texture data!\n", total_unpack_warnings);
	else
		printf("Success\n");

	return true;
}

static bool compare_mode(command_line_params &opts)
{
	if (opts.m_input_filenames.size() != 2)
	{
		error_printf("Must specify two PNG filenames using -file\n");
		return false;
	}

	image a, b;
	if (!load_png(opts.m_input_filenames[0].c_str(), a))
	{
		error_printf("Failed loading image from file \"%s\"!\n", opts.m_input_filenames[0].c_str());
		return false;
	}

	printf("Loaded \"%s\", %ux%u, has alpha: %u\n", opts.m_input_filenames[0].c_str(), a.get_width(), a.get_height(), a.has_alpha());

	if (!load_png(opts.m_input_filenames[1].c_str(), b))
	{
		error_printf("Failed loading image from file \"%s\"!\n", opts.m_input_filenames[1].c_str());
		return false;
	}

	printf("Loaded \"%s\", %ux%u, has alpha: %u\n", opts.m_input_filenames[1].c_str(), b.get_width(), b.get_height(), b.has_alpha());

	if ((a.get_width() != b.get_width()) || (a.get_height() != b.get_height()))
	{
		printf("Images don't have the same dimensions - cropping input images to smallest common dimensions\n");

		uint32_t w = minimum(a.get_width(), b.get_width());
		uint32_t h = minimum(a.get_height(), b.get_height());

		a.crop(w, h);
		b.crop(w, h);
	}

	printf("Comparison image res: %ux%u\n", a.get_width(), a.get_height());

	image_metrics im;
	im.calc(a, b, 0, 3);
	im.print("RGB    ");

	im.calc(a, b, 0, 1);
	im.print("R      ");

	im.calc(a, b, 1, 1);
	im.print("G      ");

	im.calc(a, b, 2, 1);
	im.print("B      ");

	im.calc(a, b, 0, 0);
	im.print("Y 709  " );

	im.calc(a, b, 0, 0, true, true);
	im.print("Y 601  " );
	
	if (opts.m_compare_ssim)
	{
		vec4F s_rgb(compute_ssim(a, b, false, false));

		printf("R SSIM: %f\n", s_rgb[0]);
		printf("G SSIM: %f\n", s_rgb[1]);
		printf("B SSIM: %f\n", s_rgb[2]);
		printf("RGB Avg SSIM: %f\n", (s_rgb[0] + s_rgb[1] + s_rgb[2]) / 3.0f);
		printf("A SSIM: %f\n", s_rgb[3]);

		vec4F s_y_709(compute_ssim(a, b, true, false));
		printf("Y 709 SSIM: %f\n", s_y_709[0]);

		vec4F s_y_601(compute_ssim(a, b, true, true));
		printf("Y 601 SSIM: %f\n", s_y_601[0]);
	}

	image delta_img(a.get_width(), a.get_height());

	const int X = 2;

	for (uint32_t y = 0; y < a.get_height(); y++)
	{
		for (uint32_t x = 0; x < a.get_width(); x++)
		{
			color_rgba &d = delta_img(x, y);

			for (int c = 0; c < 4; c++)
				d[c] = (uint8_t)clamp<int>((a(x, y)[c] - b(x, y)[c]) * X + 128, 0, 255);
		} // x
	} // y

	save_png("a_rgb.png", a, cImageSaveIgnoreAlpha);
	save_png("a_alpha.png", a, cImageSaveGrayscale, 3);
	printf("Wrote a_rgb.png and a_alpha.png\n");

	save_png("b_rgb.png", b, cImageSaveIgnoreAlpha);
	save_png("b_alpha.png", b, cImageSaveGrayscale, 3);
	printf("Wrote b_rgb.png and b_alpha.png\n");

	save_png("delta_img_rgb.png", delta_img, cImageSaveIgnoreAlpha);
	printf("Wrote delta_img_rgb.png\n");
	
	save_png("delta_img_a.png", delta_img, cImageSaveGrayscale, 3);
	printf("Wrote delta_img_a.png\n");
	
	return true;
}

static int main_internal(int argc, const char **argv)
{
	printf("Basis Universal GPU Texture Compressor Reference Encoder v" BASISU_TOOL_VERSION ", Copyright (C) 2019 Binomial LLC, All rights reserved\n");

	//interval_timer tm;
	//tm.start();

	basisu_encoder_init();

	//printf("Encoder and transcoder libraries initialized in %3.3f ms\n", tm.get_elapsed_ms());

#if defined(DEBUG) || defined(_DEBUG)
	printf("DEBUG build\n");
#endif

	if (argc == 1)
	{
		print_usage();
		return EXIT_FAILURE;
	}

	command_line_params opts;
	if (!opts.parse(argc, argv))
	{
		print_usage();
		return EXIT_FAILURE;
	}

	if (!opts.process_listing_files())
		return EXIT_FAILURE;

	if (opts.m_mode == cDefault)
	{
		for (size_t i = 0; i < opts.m_input_filenames.size(); i++)
		{
			std::string ext(string_get_extension(opts.m_input_filenames[i]));
			if (strcasecmp(ext.c_str(), "basis") == 0)
			{
				// If they haven't specified any modes, and they give us a .basis file, then assume they want to unpack it.
				opts.m_mode = cUnpack;
				break;
			}
		}
	}

	bool status = false;

	switch (opts.m_mode)
	{
	case cDefault:
	case cCompress:
		status = compress_mode(opts);
		break;
	case cValidate:
		status = unpack_and_validate_mode(opts, true);
		break;
	case cUnpack:
		status = unpack_and_validate_mode(opts, false);
		break;
	case cCompare:
		status = compare_mode(opts);
		break;
	case cVersion:
		status = true; // We printed the version at the beginning of main_internal
		break;
	default:
		assert(0);
		break;
	}

	return status ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, const char **argv)
{
	int status = EXIT_FAILURE;

#if BASISU_CATCH_EXCEPTIONS
	try
	{
		 status = main_internal(argc, argv);
	}
	catch (const std::exception &exc)
	{
		 fprintf(stderr, "Fatal error: Caught exception \"%s\"\n", exc.what());
	}
	catch (...)
	{
		fprintf(stderr, "Fatal error: Uncaught exception!\n");
	}
#else
	status = main_internal(argc, argv);
#endif

	return status;
}
