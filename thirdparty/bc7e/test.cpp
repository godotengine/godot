// test.cpp - Command line example/test app
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define BC7ENC_VERSION "1.08"

#define COMPUTE_SSIM (0)

#if _OPENMP
#include <omp.h>
#endif

#include "rdo_bc_encoder.h"

using namespace utils;

static int print_usage()
{
	fprintf(stderr, "\nReads PNG files (with or without alpha channels) and packs them to BC1-5 or BC7/BPTC (the default).\nFor BC7, the tool uses either bc7enc.cpp (uses modes 1/5/6/7) or bc7e.ispc (all modes, -U option, the default if SUPPORT_BC7E was TRUE).\n");
	fprintf(stderr, "Supports optional reduced entropy BC7 encoding (using -e, bc7enc.cpp only) and Rate Distortion\nOptimization (RDO - all encoders) for BC1-7 using -z# where # is lambda. Higher lambdas=more compressible files, but lower quality. Can also combine RDO with -e.\n");
	fprintf(stderr, "By default, this tool compresses to BC7. A DX10 DDS file and a unpacked PNG file will be written\nto the current directory with the .dds/_unpacked.png/_unpacked_alpha.png suffixes.\n");
	fprintf(stderr, "This tool does not yet support generating mipmaps (yet).\n");
	fprintf(stderr, "\nUsage: bc7enc [-apng_filename] [options] input_filename.png [compressed_output.dds] [unpacked_output.png]\n\n");
	fprintf(stderr, "-q Quiet mode (less debug/status output)\n");
	fprintf(stderr, "-apng_filename Load G channel of PNG file into alpha channel of source image\n");
	fprintf(stderr, "-g Don't write unpacked output PNG files (this disables PSNR metrics too).\n");
	fprintf(stderr, "-y Flip source image along Y axis before packing.\n");
	fprintf(stderr, "-o Write output files to the source file's directory, instead of the current directory.\n");
	fprintf(stderr, "-1 Encode to BC1. Use -L# option to set the base BC1 encoder's quality (default is 18 - highest quality).\n");
	fprintf(stderr, "-3 Encode to BC3. Use -L# option to set the base BC1 encoder's quality (default is 18 - highest quality).\n");
	fprintf(stderr, "-4 Encode to BC4. Use -hl and -hr# options to change the base encoder's quality.\n");
	fprintf(stderr, "-5 Encode to BC5. Use -hl and -hr# options to change the base encoder's quality.\n");
	fprintf(stderr, "-f Force writing DX10-style DDS files (otherwise for BC1-5 it uses DX9-style DDS files)\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "-X# BC4/5: Set first color channel (defaults to 0 or red)\n");
	fprintf(stderr, "-Y# BC4/5: Set second color channel (defaults to 1 or green)\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "-U BC7: Use bc7e.ispc (this is the default if it's been compiled in) instead of bc7enc.cpp. Higher quality using all BC7 modes, but doesn't support -e.\n");
	fprintf(stderr, "-C BC7: Use bc7enc.cpp instead of the default bc7e.ispc. Only supports modes 1/5/6/7. May be a slight win at certain lambdas combined with -e, but overall is weaker vs. bc7e.ispc.\n");
	fprintf(stderr, "-s BC7: Use perceptual colorspace metrics instead of linear (bc7e.ispc only). The default for all formats is to use linear RGB/RGBA metrics. RDO mode is currently always linear.\n");
	fprintf(stderr, "-uX BC7: Set the BC7 base encoder quality level. X ranges from [0,4] for bc7enc.cpp or [0,6] for bc7e.ispc. Default is 6 (highest quality).\n");
	fprintf(stderr, "-pX BC7: (bc7enc.cpp only) Scan X partitions in mode 1, X ranges from [0,64], use 0 to disable mode 1 entirely (faster)\n");
	fprintf(stderr, "-LX BC1: Set rgbcx.cpp's BC1/BC3 RGB encoding level, where 0=fastest and 18=slowest but highest quality. Default is 18.\n");
	fprintf(stderr, "\nBC3-5 alpha block encoding options:\n");
	fprintf(stderr, "-hl BC3-5: Use lower quality BC4 block encoder (much faster, but lower quality, only uses 8 value mode)\n");
	fprintf(stderr, "-h6 BC3-5: Use 6 value mode only for BC4 blocks\n");
	fprintf(stderr, "-h8 BC3-5: Use 8 value mode only for BC4 blocks\n");
	fprintf(stderr, "-hr# BC3-5: Set search radius, default is 5, larger=higher quality but slower compression\n");
	fprintf(stderr, "\nRDO encoding options (compatible with all formats/encoders):\n");
	fprintf(stderr, "-e BC7: (bc7enc.cpp only) Quantize/weight BC7 output for lower entropy (no slowdown but only 5-10%% gains, can be combined with -z# for more gains)\n");
	fprintf(stderr, "-z# BC1-7: Set RDO lambda factor (quality), lower=higher quality/larger LZ compressed files, try .1-4, combine with -e for BC7 for more gains\n");
	fprintf(stderr, "-zb# BC1-7: Manually set smooth block scale factor, higher values = less distortion on smooth blocks, try 5-70\n");
	fprintf(stderr, "-zc# BC1: Set RDO lookback window size in bytes (higher=more effective but slower, default=128, try 8-16384)\n");
	fprintf(stderr, "-zn BC1-7: Inject up to 2 matches into each block vs. 1 (the default, a little slower, but noticeably higher compression)\n");
	fprintf(stderr, "-ze BC1-7: Inject up to 1 matches into each block vs. 1 (a little faster, but less compression)\n");
	fprintf(stderr, "-zm BC1-7: Allow byte sequences to be moved inside blocks (significantly slower, not worth it in benchmarking, will likely be removed)\n");
	fprintf(stderr, "-zu BC1/3/7: Disable RGB ultrasmooth block detection/handling\n");
	fprintf(stderr, "\nRDO debugging/development:\n");
	fprintf(stderr, "-zd BC1-7: Enable RDO debug output\n");
	fprintf(stderr, "-zt BC1-7: Disable RDO multithreading\n");
	fprintf(stderr, "\n");
	fprintf(stderr, "BC1/BC3 RGB specific options:\n");
	fprintf(stderr, "-b BC1: Don't use 3-color mode transparent texels on blocks containing black or very dark pixels. By default this mode is now enabled.\n");
	fprintf(stderr, "-c BC1: Disable 3-color mode\n");
	fprintf(stderr, "-n BC1/BC3: Encode/decode BC1 for NVidia GPU's\n");
	fprintf(stderr, "-m BC1/BC3: Encode/decode BC1 for AMD GPU's\n");
	fprintf(stderr, "-r BC1/BC3: Encode/decode BC1 using ideal BC1 formulas with rounding for 4-color block colors 2,3 (same as AMD Compressonator)\n");
	fprintf(stderr, "\nBy default, this tool encodes to BC1 *without rounding* 4-color block colors 2,3, which may not match the output of some software decoders.\n");
	fprintf(stderr, "\nFor BC4 and BC5: Not all tools support reading DX9-style BC4/BC5 format files (or BC4/5 files at all). AMD Compressonator does.\n");
	fprintf(stderr, "\nFor BC1, the engine/shader must ignore decoded texture alpha because the encoder utilizes transparent texel to get black/dark texels. Use -b to disable.\n");
	fprintf(stderr, "\nReduced entropy/RDO encoding examples:\n");
	fprintf(stderr, "\n\"bc7enc -C -e blah.png\" - Reduced entropy BC7 encoding using bc7enc.cpp (extremely fast, but only 5-10%% gains, and only 4 BC7 modes)\n");
	fprintf(stderr, "\"bc7enc -z1.0 -zc32 -ze blah.png\" - RDO BC7 with lambda 1.0, window size 32 bytes (default window is 128), 1 matches per block for faster compression\n");
	fprintf(stderr, "\"bc7enc -z1.0 -zc256 blah.png\" - RDO BC7 with lambda 1.0, window size 256 bytes (default window is only 128), 2 matches per block for higher compression\n");
	fprintf(stderr, "\"bc7enc -z1.0 -C -e -zc1024 blah.png\" - RDO BC7 with lambda 1.0, window size 1024 bytes for more gains (but slower), combined with reduced entropy BC7\n");
	fprintf(stderr, "\"bc7enc -1 -z1.0 blah.png\" - RDO BC1 with lambda 1.0\n");

#if SUPPORT_BC7E
	fprintf(stderr, "\nbc7e.ispc (-U option) is supported in this build.\n");
#else
	fprintf(stderr, "\nbc7e.ispc (-U option) is NOT supported in this build.\n");
#endif			

	return EXIT_FAILURE;
}

static bool load_listing_file(const std::string& f, std::vector<std::string>& filenames)
{
	std::string filename(f);
	//filename.erase(0, 1);

	FILE* pFile = nullptr;
#ifdef _WIN32
	fopen_s(&pFile, filename.c_str(), "r");
#else
	pFile = fopen(filename.c_str(), "r");
#endif

	if (!pFile)
	{
		fprintf(stderr, "Failed opening listing file: \"%s\"\n", filename.c_str());
		return false;
	}

	uint32_t total_filenames = 0;

	for (; ; )
	{
		char buf[3072];
		buf[0] = '\0';

		char* p = fgets(buf, sizeof(buf), pFile);
		if (!p)
		{
			if (ferror(pFile))
			{
				fprintf(stderr, "Failed reading from listing file: \"%s\"\n", filename.c_str());

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

static int graph_mode(const std::string& graph_listing_file, rdo_bc::rdo_bc_params rp)
{
	rp.m_status_output = false;

	std::vector<std::string> filenames;
	if (!load_listing_file(graph_listing_file, filenames))
	{
		fprintf(stderr, "Failed loading graph listing file \"%s\"\n", graph_listing_file.c_str());
		return EXIT_FAILURE;
	}

	if (!filenames.size())
	{
		fprintf(stderr, "No files to process!\n");
		return EXIT_FAILURE;
	}

	struct encode_results
	{
		float m_lambda;
		float m_rate;
		float m_distortion;
	};

	std::vector< std::vector<encode_results> > results(filenames.size());

	static float s_lambdas[] = { .01f, .1f, .25f, .4f, .5f, .6f, .75f, .9f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
	const uint32_t TOTAL_LAMBDAS = ARRAY_SIZE(s_lambdas);

	for (uint32_t file_index = 0; file_index < filenames.size(); file_index++)
	{
		image_u8 source_image;
		if (!load_png(filenames[file_index].c_str(), source_image))
			return EXIT_FAILURE;

		printf("Source image: %s %ux%u\n", filenames[file_index].c_str(), source_image.width(), source_image.height());
				
		for (uint32_t lambda_index = 0; lambda_index < TOTAL_LAMBDAS; lambda_index++)
		{
			rp.m_rdo_lambda = s_lambdas[lambda_index];

			printf("Using lambda %f\n", rp.m_rdo_lambda);

			rdo_bc::rdo_bc_encoder encoder;
			if (!encoder.init(source_image, rp))
			{
				fprintf(stderr, "rdo_bc_encoder::init() failed!\n");
				return EXIT_FAILURE;
			}

			if (lambda_index == 0)
			{
				if (encoder.get_has_alpha())
					printf("Source image has an alpha channel.\n");
				else
					printf("Source image is opaque.\n");
			}

			if (!encoder.encode())
			{
				fprintf(stderr, "rdo_bc_encoder::encode() failed!\n");
				return EXIT_FAILURE;
			}
												
			// Compress the output data losslessly using Deflate
			const uint32_t output_data_size = encoder.get_total_blocks_size_in_bytes();
			const uint32_t pre_rdo_comp_size = get_deflate_size(encoder.get_prerdo_blocks(), output_data_size);
			const uint32_t comp_size = get_deflate_size(encoder.get_blocks(), output_data_size);

			float pre_rdo_lz_bits_per_texel = (pre_rdo_comp_size * 8.0f) / encoder.get_total_texels();
			UNUSED(pre_rdo_lz_bits_per_texel);
			float lz_bits_per_texel = comp_size * 8.0f / encoder.get_total_texels();

			image_u8 unpacked_image;
			if (!encoder.unpack_blocks(unpacked_image))
				return EXIT_FAILURE;

			image_metrics rgb_metrics;
			rgb_metrics.compute(source_image, unpacked_image, 0, 3);

			results[file_index].resize(lambda_index + 1);
			results[file_index][lambda_index].m_distortion = (float)rgb_metrics.m_peak_snr;
			results[file_index][lambda_index].m_rate = lz_bits_per_texel;
			results[file_index][lambda_index].m_lambda = rp.m_rdo_lambda;

			printf("Rate %f distortion %f\n", lz_bits_per_texel, rgb_metrics.m_peak_snr);

		} // lambda_index

		printf("\n");

	} // file_index

	FILE* pCSV_file = fopen("graph5.csv", "w");
	if (!pCSV_file)
	{
		fprintf(stderr, "Failed creating graph.csv!\n");
		return EXIT_FAILURE;
	}

	fprintf(pCSV_file, "lambda, avg_rate, avg_psnr,");
	
	for (uint32_t file_index = 0; file_index < filenames.size(); file_index++)
	{
		std::string fname;
		fname = filenames[file_index].c_str();
		strip_path(fname);
		fprintf(pCSV_file, "%s Rate, %s PSNR%c", fname.c_str(), fname.c_str(), file_index != (filenames.size() - 1) ? ',' : ' ');
	}
	fprintf(pCSV_file, "\n");
		
	for (uint32_t lambda_index = 0; lambda_index < TOTAL_LAMBDAS; lambda_index++)
	{
		fprintf(pCSV_file, "%f,", s_lambdas[lambda_index]);

		float total_rate = 0.0f, total_distortion = 0.0f;
		for (uint32_t file_index = 0; file_index < filenames.size(); file_index++)
		{
			total_rate += results[file_index][lambda_index].m_rate;
			total_distortion += results[file_index][lambda_index].m_distortion;
		}
		total_rate /= filenames.size();
		total_distortion /= filenames.size();
		
		fprintf(pCSV_file, "%f,%f,", total_rate, total_distortion);

		for (uint32_t file_index = 0; file_index < filenames.size(); file_index++)
		{
			fprintf(pCSV_file, "%f,%f", results[file_index][lambda_index].m_rate, results[file_index][lambda_index].m_distortion);
			if (file_index != (filenames.size() - 1))
				fprintf(pCSV_file, ",");
		}
		fprintf(pCSV_file, "\n");
	}

	fclose(pCSV_file);
	pCSV_file = nullptr;

	return EXIT_SUCCESS;
}

int main(int argc, char* argv[])
{
	bool quiet_mode = false;

	for (int i = 0; i < argc; i++)
		if (strcmp(argv[i], "-q") == 0)
			quiet_mode = true;

	if (!quiet_mode)
		printf("bc7enc v%s - RDO BC1-7 Texture Compressor\n", BC7ENC_VERSION);

	int max_threads = 1;
#if _OPENMP
	max_threads = std::min(std::max(1, omp_get_max_threads()), 128);
#endif
		
	if (argc < 2)
		return print_usage();
		
	std::string src_filename, src_alpha_filename, dds_output_filename, png_output_filename, png_alpha_output_filename;

	bool no_output_png = false;
	bool out_cur_dir = true;
	bool force_dx10_dds = false;
	FILE* pCSV_file = nullptr;

	uint32_t pixel_format_bpp = 8;

	rdo_bc::rdo_bc_params rp;
	rp.m_rdo_max_threads = max_threads;
	rp.m_status_output = !quiet_mode;

	std::string graph_listing_file;

	for (int i = 1; i < argc; i++)
	{
		const char *pArg = argv[i];
		if (pArg[0] == '-')
		{
			switch (pArg[1])
			{
				case 'G':
				{
					graph_listing_file = (pArg + 2);
					break;
				}
				case 'q':
				{
					break;
				}
				case 'U':
				{
					rp.m_use_bc7e = true;
					break;
				}
				case 'C':
				{
					rp.m_use_bc7e = false;
					break;
				}
				case 'e':
				{
					rp.m_bc7enc_reduce_entropy = true;
					break;
				}
				case 'h':
				{
					if (strcmp(pArg, "-hl") == 0)
						rp.m_use_hq_bc345 = false;
					else if (strcmp(pArg, "-h6") == 0)
						rp.m_bc345_mode_mask = rgbcx::BC4_USE_MODE6_FLAG;
					else if (strcmp(pArg, "-h8") == 0)
						rp.m_bc345_mode_mask = rgbcx::BC4_USE_MODE8_FLAG;
					else if (strncmp(pArg, "-hr", 3) == 0)
					{
						rp.m_bc345_search_rad = atoi(pArg + 3);
						rp.m_bc345_search_rad = std::max(0, std::min(32, rp.m_bc345_search_rad));
					}
											
					break;
				}
				case '6':
				{
					rp.m_bc7enc_mode6_only = true;
					break;
				}
				case '1':
				{
					rp.m_dxgi_format = DXGI_FORMAT_BC1_UNORM;
					pixel_format_bpp = 4;
					printf("Compressing to BC1\n");
					break;
				}
				case '3':
				{
					rp.m_dxgi_format = DXGI_FORMAT_BC3_UNORM;
					pixel_format_bpp = 8;
					printf("Compressing to BC3\n");
					break;
				}
				case '4':
				{
					rp.m_dxgi_format = DXGI_FORMAT_BC4_UNORM;
					pixel_format_bpp = 4;
					printf("Compressing to BC4\n");
					break;
				}
				case '5':
				{
					rp.m_dxgi_format = DXGI_FORMAT_BC5_UNORM;
					pixel_format_bpp = 8;
					printf("Compressing to BC5\n");
					break;
				}
				case 'y':
				{
					rp.m_y_flip = true;
					break;
				}
				case 'a':
				{
					src_alpha_filename = pArg + 2;
					break;
				}
				case 'X':
				{
					rp.m_bc45_channel0 = atoi(pArg + 2);
					if ((rp.m_bc45_channel0 < 0) || (rp.m_bc45_channel0 > 3))
					{
						fprintf(stderr, "Invalid argument: %s\n", pArg);
						return EXIT_FAILURE;
					}
					break;
				}
				case 'Y':
				{
					rp.m_bc45_channel1 = atoi(pArg + 2);
					if ((rp.m_bc45_channel1 < 0) || (rp.m_bc45_channel1 > 3))
					{
						fprintf(stderr, "Invalid argument: %s\n", pArg);
						return EXIT_FAILURE;
					}
					break;
				}
				case 'f':
				{
					force_dx10_dds = true;
					break;
				}
				case 'u':
				{
					rp.m_bc7_uber_level = atoi(pArg + 2);
					if ((rp.m_bc7_uber_level < 0) || (rp.m_bc7_uber_level > 6)) //BC7ENC_MAX_UBER_LEVEL))
					{
						fprintf(stderr, "Invalid argument: %s\n", pArg);
						return EXIT_FAILURE;
					}
					break;

				}
				case 'L':
				{
					rp.m_bc1_quality_level = atoi(pArg + 2);
					if (((int)rp.m_bc1_quality_level < (int)rgbcx::MIN_LEVEL) || ((int)rp.m_bc1_quality_level > (int)(rgbcx::MAX_LEVEL + 1)))
					{
						fprintf(stderr, "Invalid argument: %s\n", pArg);
						return EXIT_FAILURE;
					}
					break;

				}
				case 'g':
				{
					no_output_png = true;
					break;
				}
				case 's':
				{
					rp.m_perceptual = true;
					break;
				}
				case 'p':
				{
					rp.m_bc7enc_max_partitions_to_scan = atoi(pArg + 2);
					if ((rp.m_bc7enc_max_partitions_to_scan < 0) || (rp.m_bc7enc_max_partitions_to_scan > BC7ENC_MAX_PARTITIONS))
					{
						fprintf(stderr, "Invalid argument: %s\n", pArg);
						return EXIT_FAILURE;
					}
					break;
				}
				case 'n':
				{
					rp.m_bc1_mode = rgbcx::bc1_approx_mode::cBC1NVidia;
					break;
				}
				case 'm':
				{
					rp.m_bc1_mode = rgbcx::bc1_approx_mode::cBC1AMD;
					break;
				}
				case 'r':
				{
					rp.m_bc1_mode = rgbcx::bc1_approx_mode::cBC1IdealRound4;
					break;
				}
				case 'z':
				{
					if (strncmp(pArg, "-zt", 3) == 0)
					{
						rp.m_rdo_multithreading = false;
					}
					else if (strncmp(pArg, "-zd", 3) == 0)
					{
						rp.m_rdo_debug_output = true;
					}
					else if (strncmp(pArg, "-zq", 3) == 0)
					{
						rp.m_bc7enc_rdo_bc7_quant_mode6_endpoints = false;
					}
					else if (strncmp(pArg, "-zw", 3) == 0)
					{
						rp.m_bc7enc_rdo_bc7_weight_modes = false;
					}
					else if (strncmp(pArg, "-zp", 3) == 0)
					{
						rp.m_bc7enc_rdo_bc7_weight_low_frequency_partitions = false;
					}
					else if (strncmp(pArg, "-zo", 3) == 0)
					{
						rp.m_bc7enc_rdo_bc7_pbit1_weighting = false;
					}
					else if (strncmp(pArg, "-zm", 3) == 0)
					{
						rp.m_rdo_allow_relative_movement = true;
					}
					else if (strncmp(pArg, "-zn", 3) == 0)
					{
						rp.m_rdo_try_2_matches = true;
					}
					else if (strncmp(pArg, "-ze", 3) == 0)
					{
						rp.m_rdo_try_2_matches = false;
					}
					else if (strncmp(pArg, "-zu", 3) == 0)
					{
						rp.m_rdo_ultrasmooth_block_handling = false;
					}
					else if (strncmp(pArg, "-zb", 3) == 0)
					{
						rp.m_rdo_smooth_block_error_scale = (float)atof(pArg + 3);
						rp.m_rdo_smooth_block_error_scale = std::min<float>(std::max<float>(rp.m_rdo_smooth_block_error_scale, 1.0f), 500.0f);
						rp.m_custom_rdo_smooth_block_error_scale = true;
					}
					else if (strncmp(pArg, "-zc", 3) == 0)
					{
						rp.m_lookback_window_size = atoi(pArg + 3);
						rp.m_lookback_window_size = std::min<int>(std::max<int>(rp.m_lookback_window_size, 8), 65536*2);
						rp.m_custom_lookback_window_size = true;
					}
					else if (strncmp(pArg, "-zv", 3) == 0)
					{
						rp.m_rdo_max_smooth_block_std_dev = (float)atof(pArg + 3);
						rp.m_rdo_max_smooth_block_std_dev = std::min<float>(std::max<float>(rp.m_rdo_max_smooth_block_std_dev, .000125f), 256.0f);
					}
					else
					{
						rp.m_rdo_lambda = (float)atof(pArg + 2);
						rp.m_rdo_lambda = std::min<float>(std::max<float>(rp.m_rdo_lambda, 0.0f), 500.0f);
					}
					break;
				}
				case 'o':
				{
					out_cur_dir = false;
					break;
				}
				case 'b':
				{
					rp.m_use_bc1_3color_mode_for_black = false;
					break;
				}
				case 'c':
				{
					rp.m_use_bc1_3color_mode = false;
					break;
				}
				case 'v':
				{
					if (pCSV_file)
						fclose(pCSV_file);
					
					pCSV_file = fopen(pArg + 2, "a");
					if (!pCSV_file)
					{
						fprintf(stderr, "Failed opening file %s\n", pArg + 2);
						return EXIT_FAILURE;
					}
					break;
				}
				default:
				{
					fprintf(stderr, "Invalid argument: %s\n", pArg);
					return EXIT_FAILURE;
				}
			}
		}
		else
		{
			if (!src_filename.size())
				src_filename = pArg;
			else if (!dds_output_filename.size())
				dds_output_filename = pArg;
			else if (!png_output_filename.size())
				png_output_filename = pArg;
			else
			{
				fprintf(stderr, "Invalid argument: %s\n", pArg);
				return EXIT_FAILURE;
			}
		}
	}

	if (rp.m_status_output)
	{
		printf("Max threads: %u\n", max_threads);
		printf("Supports bc7e.ispc: %u\n", SUPPORT_BC7E);
	}

	if (graph_listing_file.size())
	{
		return graph_mode(graph_listing_file, rp);
	}
		
	if (!src_filename.size())
	{
		fprintf(stderr, "No source filename specified!\n");
		return EXIT_FAILURE;
	}

	if (!dds_output_filename.size())
	{
		dds_output_filename = src_filename;
		strip_extension(dds_output_filename);
		if (out_cur_dir)
			strip_path(dds_output_filename);
		dds_output_filename += ".dds";
	}

	if (!png_output_filename.size())
	{
		png_output_filename = src_filename;
		strip_extension(png_output_filename);
		if (out_cur_dir)
			strip_path(png_output_filename);
		png_output_filename += "_unpacked.png";
	}

	png_alpha_output_filename = png_output_filename;
	strip_extension(png_alpha_output_filename);
	png_alpha_output_filename += "_alpha.png";
				
	image_u8 source_image;
	if (!load_png(src_filename.c_str(), source_image))
		return EXIT_FAILURE;
	
	if (rp.m_status_output)
		printf("Source image: %s %ux%u\n", src_filename.c_str(), source_image.width(), source_image.height());

	if (src_alpha_filename.size())
	{
		image_u8 source_alpha_image;
		if (!load_png(src_alpha_filename.c_str(), source_alpha_image))
			return EXIT_FAILURE;

		if (rp.m_status_output)
			printf("Source alpha image: %s %ux%u\n", src_alpha_filename.c_str(), source_alpha_image.width(), source_alpha_image.height());

		const uint32_t w = std::min(source_alpha_image.width(), source_image.width());
		const uint32_t h = std::min(source_alpha_image.height(), source_image.height());
		
		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++)
				source_image(x, y)[3] = source_alpha_image(x, y)[1];
	}

	clock_t overall_start_t = clock();

	rdo_bc::rdo_bc_encoder encoder;
	if (!encoder.init(source_image, rp))
	{
		fprintf(stderr, "rdo_bc_encoder::init() failed!\n");
		return EXIT_FAILURE;
	}

	if (rp.m_status_output)
	{
		if (encoder.get_has_alpha())
			printf("Source image has an alpha channel.\n");
		else
			printf("Source image is opaque.\n");
	}

	if (!encoder.encode())
	{
		fprintf(stderr, "rdo_bc_encoder::encode() failed!\n");
		return EXIT_FAILURE;
	}

	clock_t overall_end_t = clock();

	if (rp.m_status_output)
		printf("Total processing time: %f secs\n", (double)(overall_end_t - overall_start_t) / CLOCKS_PER_SEC);

	// Compress the output data losslessly using Deflate
	const uint32_t output_data_size = encoder.get_total_blocks_size_in_bytes();
	const uint32_t pre_rdo_comp_size = get_deflate_size(encoder.get_prerdo_blocks(), output_data_size);

	float pre_rdo_lz_bits_per_texel = (pre_rdo_comp_size * 8.0f) / encoder.get_total_texels();

	if (rp.m_status_output)
	{
		printf("Output data size: %u, LZ (Deflate) compressed file size: %u, %3.2f bits/texel\n",
			output_data_size,
			(uint32_t)pre_rdo_comp_size,
			pre_rdo_lz_bits_per_texel);
	}
			
	const uint32_t comp_size = get_deflate_size(encoder.get_blocks(), output_data_size);
		
	float lz_bits_per_texel = comp_size * 8.0f / encoder.get_total_texels();

	if (rp.m_status_output)
		printf("RDO output data size: %u, LZ (Deflate) compressed file size: %u, %3.2f bits/texel, savings: %3.2f%%\n", output_data_size, (uint32_t)comp_size, lz_bits_per_texel, 
			(lz_bits_per_texel != pre_rdo_lz_bits_per_texel) ? 100.0f - (lz_bits_per_texel * 100.0f) / pre_rdo_lz_bits_per_texel : 0.0f);
			
	if (!save_dds(dds_output_filename.c_str(), encoder.get_orig_width(), encoder.get_orig_height(),
		encoder.get_blocks(), pixel_format_bpp, rp.m_dxgi_format, rp.m_perceptual, force_dx10_dds))
	{
		fprintf(stderr, "Failed writing file \"%s\"\n", dds_output_filename.c_str());
		return EXIT_FAILURE;
	}
	
	if (rp.m_status_output)
		printf("Wrote DDS file %s\n", dds_output_filename.c_str());

	float csv_psnr = 0.0f, csv_ssim = 0.0f;
	(void)csv_ssim;

	if ((!no_output_png) && (png_output_filename.size()))
	{
		image_u8 unpacked_image;
		if (!encoder.unpack_blocks(unpacked_image))
			return EXIT_FAILURE;

		if ((rp.m_dxgi_format != DXGI_FORMAT_BC4_UNORM) && (rp.m_dxgi_format != DXGI_FORMAT_BC5_UNORM))
		{
			image_metrics y_metrics;
			y_metrics.compute(source_image, unpacked_image, 0, 0);
			if (rp.m_status_output)
				printf("Luma  Max error: %3.0f RMSE: %f PSNR %03.3f dB, PSNR per bits/texel: %f\n", y_metrics.m_max, y_metrics.m_root_mean_squared, y_metrics.m_peak_snr, y_metrics.m_peak_snr / lz_bits_per_texel * 10000.0f);

			image_metrics rgb_metrics;
			rgb_metrics.compute(source_image, unpacked_image, 0, 3);
			if (rp.m_status_output)
				printf("RGB   Max error: %3.0f RMSE: %f PSNR %03.3f dB, PSNR per bits/texel: %f\n", rgb_metrics.m_max, rgb_metrics.m_root_mean_squared, rgb_metrics.m_peak_snr, rgb_metrics.m_peak_snr / lz_bits_per_texel * 10000.0f);

			csv_psnr = (float)rgb_metrics.m_peak_snr;

			image_metrics rgba_metrics;
			rgba_metrics.compute(source_image, unpacked_image, 0, 4);
			if (rp.m_status_output)
				printf("RGBA  Max error: %3.0f RMSE: %f PSNR %03.3f dB, PSNR per bits/texel: %f\n", rgba_metrics.m_max, rgba_metrics.m_root_mean_squared, rgba_metrics.m_peak_snr, rgba_metrics.m_peak_snr / lz_bits_per_texel * 10000.0f);

#if COMPUTE_SSIM
			vec4F ssim_y(compute_ssim(source_image, unpacked_image, true));
			vec4F ssim_rgba(compute_ssim(source_image, unpacked_image, false));

			if (rp.m_status_output)
			{
				printf("R       SSIM: %f\n", ssim_rgba[0]);
				printf("G       SSIM: %f\n", ssim_rgba[1]);
				printf("B       SSIM: %f\n", ssim_rgba[2]);
				printf("RGB Avg SSIM: %f\n", (ssim_rgba[0] + ssim_rgba[1] + ssim_rgba[2]) / 3.0f);
				printf("A       SSIM: %f\n", ssim_rgba[3]);

				printf("Luma    SSIM: %f\n", ssim_y[0]);
			}

			csv_ssim = (ssim_rgba[0] + ssim_rgba[1] + ssim_rgba[2]) / 3.0f;
#endif
		}
						
		for (uint32_t chan = 0; chan < 4; chan++)
		{
			if (rp.m_dxgi_format == DXGI_FORMAT_BC4_UNORM)
			{
				if (chan != rp.m_bc45_channel0)
					continue;
			}
			else if (rp.m_dxgi_format == DXGI_FORMAT_BC5_UNORM)
			{
				if ((chan != rp.m_bc45_channel0) && (chan != rp.m_bc45_channel1))
					continue;
			}

			image_metrics c_metrics;
			c_metrics.compute(source_image, unpacked_image, chan, 1);
			static const char *s_chan_names[4] = { "Red  ", "Green", "Blue ", "Alpha" };
			
			if (rp.m_status_output)
				printf("%s Max error: %3.0f RMSE: %f PSNR %03.3f dB\n", s_chan_names[chan], c_metrics.m_max, c_metrics.m_root_mean_squared, c_metrics.m_peak_snr);

			if (rp.m_dxgi_format == DXGI_FORMAT_BC4_UNORM)
				csv_psnr = (float)c_metrics.m_peak_snr;
		}

		if (rp.m_dxgi_format == DXGI_FORMAT_BC5_UNORM)
		{
			image_metrics c_metrics;
			c_metrics.compute(source_image, unpacked_image, 0, 2);
			
			if (rp.m_status_output)
				printf("RG Max error: %3.0f RMSE: %f PSNR %03.3f dB\n", c_metrics.m_max, c_metrics.m_root_mean_squared, c_metrics.m_peak_snr);

			csv_psnr = (float)c_metrics.m_peak_snr;
		}

		if (rp.m_status_output)
		{
			if (rp.m_bc1_mode != rgbcx::bc1_approx_mode::cBC1Ideal)
				printf("Note: BC1/BC3 RGB decoding was done with the specified vendor's BC1 approximations.\n");
		}

		image_u8 unpacked_image_cropped(unpacked_image);
		unpacked_image_cropped.crop(encoder.get_orig_width(), encoder.get_orig_height());
		if (!save_png(png_output_filename.c_str(), unpacked_image_cropped, false))
		{
			fprintf(stderr, "Failed writing file \"%s\"\n", png_output_filename.c_str());
			return EXIT_FAILURE;
		}

		if (rp.m_status_output)
			printf("Wrote PNG file %s\n", png_output_filename.c_str());
				
		if (png_alpha_output_filename.size())
		{
			image_u8 unpacked_image_alpha(unpacked_image);
			for (uint32_t y = 0; y < unpacked_image_alpha.height(); y++)
				for (uint32_t x = 0; x < unpacked_image_alpha.width(); x++)
					unpacked_image_alpha(x, y).set(unpacked_image_alpha(x, y)[3], 255);
			unpacked_image_alpha.crop(encoder.get_orig_width(), encoder.get_orig_height());

			if (!save_png(png_alpha_output_filename.c_str(), unpacked_image_alpha, false))
			{
				fprintf(stderr, "Failed writing file \"%s\"\n", png_alpha_output_filename.c_str());
				return EXIT_FAILURE;
			}
			
			if (rp.m_status_output)
				printf("Wrote PNG file %s\n", png_alpha_output_filename.c_str());
		}
	}

	if (pCSV_file)
	{
		fprintf(pCSV_file, "%f,%f,%f\n", rp.m_rdo_lambda, lz_bits_per_texel, csv_psnr);
		fclose(pCSV_file);
		pCSV_file = nullptr;
	}

	return EXIT_SUCCESS;
}
