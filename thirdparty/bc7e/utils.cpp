// File: utils.cpp
#include "utils.h"
#include "lodepng.h"
#include "miniz.h"

namespace utils 
{
		
#define FLOOD_PUSH(y, xl, xr, dy) if (((y + (dy)) >= 0) && ((y + (dy)) < (int)m_height)) { stack.push_back(fill_segment(y, xl, xr, dy)); }

// See http://www.realtimerendering.com/resources/GraphicsGems/gems/SeedFill.c
uint32_t image_u8::flood_fill(int x, int y, const color_quad_u8& c, const color_quad_u8& b, std::vector<pixel_coord>* pSet_pixels)
{
	uint32_t total_set = 0;

	if (!flood_fill_is_inside(x, y, b))
		return 0;

	std::vector<fill_segment> stack;
	stack.reserve(64);

	FLOOD_PUSH(y, x, x, 1);
	FLOOD_PUSH(y + 1, x, x, -1);

	while (stack.size())
	{
		fill_segment s = stack.back();
		stack.pop_back();

		int x1 = s.m_xl, x2 = s.m_xr, dy = s.m_dy;
		y = s.m_y + s.m_dy;

		for (x = x1; (x >= 0) && flood_fill_is_inside(x, y, b); x--)
		{
			(*this)(x, y) = c;
			total_set++;
			if (pSet_pixels)
				pSet_pixels->push_back(pixel_coord(x, y));
		}

		int l;

		if (x >= x1)
			goto skip;

		l = x + 1;
		if (l < x1)
			FLOOD_PUSH(y, l, x1 - 1, -dy);

		x = x1 + 1;

		do
		{
			for (; x <= ((int)m_width - 1) && flood_fill_is_inside(x, y, b); x++)
			{
				(*this)(x, y) = c;
				total_set++;
				if (pSet_pixels)
					pSet_pixels->push_back(pixel_coord(x, y));
			}
			FLOOD_PUSH(y, l, x - 1, dy);

			if (x > (x2 + 1))
				FLOOD_PUSH(y, x2 + 1, x - 1, -dy);

		skip:
			for (x++; x <= x2 && !flood_fill_is_inside(x, y, b); x++)
				;

			l = x;
		} while (x <= x2);
	}

	return total_set;
}

void image_u8::draw_line(int xs, int ys, int xe, int ye, const color_quad_u8& color)
{
	if (xs > xe)
	{
		std::swap(xs, xe);
		std::swap(ys, ye);
	}

	int dx = xe - xs, dy = ye - ys;
	if (!dx)
	{
		if (ys > ye)
			std::swap(ys, ye);
		for (int i = ys; i <= ye; i++)
			set_pixel_clipped(xs, i, color);
	}
	else if (!dy)
	{
		for (int i = xs; i < xe; i++)
			set_pixel_clipped(i, ys, color);
	}
	else if (dy > 0)
	{
		if (dy <= dx)
		{
			int e = 2 * dy - dx, e_no_inc = 2 * dy, e_inc = 2 * (dy - dx);
			rasterize_line(xs, ys, xe, ye, 0, 1, e, e_inc, e_no_inc, color);
		}
		else
		{
			int e = 2 * dx - dy, e_no_inc = 2 * dx, e_inc = 2 * (dx - dy);
			rasterize_line(xs, ys, xe, ye, 1, 1, e, e_inc, e_no_inc, color);
		}
	}
	else
	{
		dy = -dy;
		if (dy <= dx)
		{
			int e = 2 * dy - dx, e_no_inc = 2 * dy, e_inc = 2 * (dy - dx);
			rasterize_line(xs, ys, xe, ye, 0, -1, e, e_inc, e_no_inc, color);
		}
		else
		{
			int e = 2 * dx - dy, e_no_inc = (2 * dx), e_inc = 2 * (dx - dy);
			rasterize_line(xe, ye, xs, ys, 1, -1, e, e_inc, e_no_inc, color);
		}
	}
}

void image_u8::rasterize_line(int xs, int ys, int xe, int ye, int pred, int inc_dec, int e, int e_inc, int e_no_inc, const color_quad_u8& color)
{
	int start, end, var;

	if (pred)
	{
		start = ys;
		end = ye;
		var = xs;
		for (int i = start; i <= end; i++)
		{
			set_pixel_clipped(var, i, color);
			if (e < 0)
				e += e_no_inc;
			else
			{
				var += inc_dec;
				e += e_inc;
			}
		}
	}
	else
	{
		start = xs;
		end = xe;
		var = ys;
		for (int i = start; i <= end; i++)
		{
			set_pixel_clipped(i, var, color);
			if (e < 0)
				e += e_no_inc;
			else
			{
				var += inc_dec;
				e += e_inc;
			}
		}
	}
}

bool load_png(const char* pFilename, image_u8& img)
{
	img.clear();

	std::vector<unsigned char> pixels;
	unsigned int w = 0, h = 0;
	unsigned int e = lodepng::decode(pixels, w, h, pFilename);
	if (e != 0)
	{
		fprintf(stderr, "Failed loading PNG file %s\n", pFilename);
		return false;
	}

	img.init(w, h);
	memcpy(&img.get_pixels()[0], &pixels[0], w * h * sizeof(uint32_t));

	return true;
}

bool save_png(const char* pFilename, const image_u8& img, bool save_alpha)
{
	const uint32_t w = img.width();
	const uint32_t h = img.height();

	std::vector<unsigned char> pixels;
	if (save_alpha)
	{
		pixels.resize(w * h * sizeof(color_quad_u8));
		memcpy(&pixels[0], &img.get_pixels()[0], w * h * sizeof(color_quad_u8));
	}
	else
	{
		pixels.resize(w * h * 3);
		unsigned char* pDst = &pixels[0];
		for (uint32_t y = 0; y < h; y++)
			for (uint32_t x = 0; x < w; x++, pDst += 3)
				pDst[0] = img(x, y)[0], pDst[1] = img(x, y)[1], pDst[2] = img(x, y)[2];
	}

	return lodepng::encode(pFilename, pixels, w, h, save_alpha ? LCT_RGBA : LCT_RGB) == 0;
}

static float gauss(int x, int y, float sigma_sqr)
{
	float pow = expf(-((x * x + y * y) / (2.0f * sigma_sqr)));
	float g = (1.0f / (sqrtf((float)(2.0f * M_PI * sigma_sqr)))) * pow;
	return g;
}

// size_x/y should be odd
void compute_gaussian_kernel(float* pDst, int size_x, int size_y, float sigma_sqr, uint32_t flags)
{
	assert(size_x & size_y & 1);

	if (!(size_x | size_y))
		return;

	int mid_x = size_x / 2;
	int mid_y = size_y / 2;

	double sum = 0;
	for (int x = 0; x < size_x; x++)
	{
		for (int y = 0; y < size_y; y++)
		{
			float g;
			if ((x > mid_x) && (y < mid_y))
				g = pDst[(size_x - x - 1) + y * size_x];
			else if ((x < mid_x) && (y > mid_y))
				g = pDst[x + (size_y - y - 1) * size_x];
			else if ((x > mid_x) && (y > mid_y))
				g = pDst[(size_x - x - 1) + (size_y - y - 1) * size_x];
			else
				g = gauss(x - mid_x, y - mid_y, sigma_sqr);

			pDst[x + y * size_x] = g;
			sum += g;
		}
	}

	if (flags & cComputeGaussianFlagNormalizeCenterToOne)
	{
		sum = pDst[mid_x + mid_y * size_x];
	}

	if (flags & (cComputeGaussianFlagNormalizeCenterToOne | cComputeGaussianFlagNormalize))
	{
		double one_over_sum = 1.0f / sum;
		for (int i = 0; i < size_x * size_y; i++)
			pDst[i] = static_cast<float>(pDst[i] * one_over_sum);

		if (flags & cComputeGaussianFlagNormalizeCenterToOne)
			pDst[mid_x + mid_y * size_x] = 1.0f;
	}

	if (flags & cComputeGaussianFlagPrint)
	{
		printf("{\n");
		for (int y = 0; y < size_y; y++)
		{
			printf("  ");
			for (int x = 0; x < size_x; x++)
			{
				printf("%f, ", pDst[x + y * size_x]);
			}
			printf("\n");
		}
		printf("}");
	}
}

void gaussian_filter(imagef& dst, const imagef& orig_img, uint32_t odd_filter_width, float sigma_sqr, bool wrapping, uint32_t width_divisor, uint32_t height_divisor)
{
	assert(odd_filter_width && (odd_filter_width & 1));
	odd_filter_width |= 1;

	std::vector<float> kernel(odd_filter_width * odd_filter_width);
	compute_gaussian_kernel(&kernel[0], odd_filter_width, odd_filter_width, sigma_sqr, cComputeGaussianFlagNormalize);

	const int dst_width = orig_img.get_width() / width_divisor;
	const int dst_height = orig_img.get_height() / height_divisor;

	const int H = odd_filter_width / 2;
	const int L = -H;

	dst.crop(dst_width, dst_height);

#pragma omp parallel for
	for (int oy = 0; oy < dst_height; oy++)
	{
		for (int ox = 0; ox < dst_width; ox++)
		{
			vec4F c(0.0f);

			for (int yd = L; yd <= H; yd++)
			{
				int y = oy * height_divisor + (height_divisor >> 1) + yd;

				for (int xd = L; xd <= H; xd++)
				{
					int x = ox * width_divisor + (width_divisor >> 1) + xd;

					const vec4F& p = orig_img.get_clamped_or_wrapped(x, y, wrapping, wrapping);

					float w = kernel[(xd + H) + (yd + H) * odd_filter_width];
					c[0] += p[0] * w;
					c[1] += p[1] * w;
					c[2] += p[2] * w;
					c[3] += p[3] * w;
				}
			}

			dst(ox, oy).set(c[0], c[1], c[2], c[3]);
		}
	}
}

static void pow_image(const imagef& src, imagef& dst, const vec4F& power)
{
	dst.resize(src);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& p = src(x, y);

			if ((power[0] == 2.0f) && (power[1] == 2.0f) && (power[2] == 2.0f) && (power[3] == 2.0f))
				dst(x, y).set(p[0] * p[0], p[1] * p[1], p[2] * p[2], p[3] * p[3]);
			else
				dst(x, y).set(powf(p[0], power[0]), powf(p[1], power[1]), powf(p[2], power[2]), powf(p[3], power[3]));
		}
	}
}

#if 0
static void mul_image(const imagef& src, imagef& dst, const vec4F& mul)
{
	dst.resize(src);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& p = src(x, y);
			dst(x, y).set(p[0] * mul[0], p[1] * mul[1], p[2] * mul[2], p[3] * mul[3]);
		}
	}
}
#endif

static void scale_image(const imagef& src, imagef& dst, const vec4F& scale, const vec4F& shift)
{
	dst.resize(src);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& p = src(x, y);

			vec4F d;

			for (uint32_t c = 0; c < 4; c++)
				d[c] = scale[c] * p[c] + shift[c];

			dst(x, y).set(d[0], d[1], d[2], d[3]);
		}
	}
}

static void add_weighted_image(const imagef& src1, const vec4F& alpha, const imagef& src2, const vec4F& beta, const vec4F& gamma, imagef& dst)
{
	dst.resize(src1);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& s1 = src1(x, y);
			const vec4F& s2 = src2(x, y);

			dst(x, y).set(
				s1[0] * alpha[0] + s2[0] * beta[0] + gamma[0],
				s1[1] * alpha[1] + s2[1] * beta[1] + gamma[1],
				s1[2] * alpha[2] + s2[2] * beta[2] + gamma[2],
				s1[3] * alpha[3] + s2[3] * beta[3] + gamma[3]);
		}
	}
}

static void add_image(const imagef& src1, const imagef& src2, imagef& dst)
{
	dst.resize(src1);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& s1 = src1(x, y);
			const vec4F& s2 = src2(x, y);

			dst(x, y).set(s1[0] + s2[0], s1[1] + s2[1], s1[2] + s2[2], s1[3] + s2[3]);
		}
	}
}

static void adds_image(const imagef& src, const vec4F& value, imagef& dst)
{
	dst.resize(src);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& p = src(x, y);

			dst(x, y).set(p[0] + value[0], p[1] + value[1], p[2] + value[2], p[3] + value[3]);
		}
	}
}

static void mul_image(const imagef& src1, const imagef& src2, imagef& dst, const vec4F& scale)
{
	dst.resize(src1);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& s1 = src1(x, y);
			const vec4F& s2 = src2(x, y);

			vec4F d;

			for (uint32_t c = 0; c < 4; c++)
			{
				float v1 = s1[c];
				float v2 = s2[c];
				d[c] = v1 * v2 * scale[c];
			}

			dst(x, y) = d;
		}
	}
}

static void div_image(const imagef& src1, const imagef& src2, imagef& dst, const vec4F& scale)
{
	dst.resize(src1);

#pragma omp parallel for
	for (int y = 0; y < (int)dst.get_height(); y++)
	{
		for (uint32_t x = 0; x < dst.get_width(); x++)
		{
			const vec4F& s1 = src1(x, y);
			const vec4F& s2 = src2(x, y);

			vec4F d;

			for (uint32_t c = 0; c < 4; c++)
			{
				float v = s2[c];
				if (v == 0.0f)
					d[c] = 0.0f;
				else
					d[c] = (s1[c] * scale[c]) / v;
			}

			dst(x, y) = d;
		}
	}
}

static vec4F avg_image(const imagef& src)
{
	vec4F avg(0.0f);

	for (uint32_t y = 0; y < src.get_height(); y++)
	{
		for (uint32_t x = 0; x < src.get_width(); x++)
		{
			const vec4F& s = src(x, y);

			avg += vec4F(s[0], s[1], s[2], s[3]);
		}
	}

	avg /= static_cast<float>(src.get_total_pixels());

	return avg;
}

// Reference: https://ece.uwaterloo.ca/~z70wang/research/ssim/index.html
vec4F compute_ssim(const imagef& a, const imagef& b)
{
	imagef axb, a_sq, b_sq, mu1, mu2, mu1_sq, mu2_sq, mu1_mu2, s1_sq, s2_sq, s12, smap, t1, t2, t3;

	const float C1 = 6.50250f, C2 = 58.52250f;

	pow_image(a, a_sq, vec4F(2));
	pow_image(b, b_sq, vec4F(2));
	mul_image(a, b, axb, vec4F(1.0f));

	gaussian_filter(mu1, a, 11, 1.5f * 1.5f);
	gaussian_filter(mu2, b, 11, 1.5f * 1.5f);

	pow_image(mu1, mu1_sq, vec4F(2));
	pow_image(mu2, mu2_sq, vec4F(2));
	mul_image(mu1, mu2, mu1_mu2, vec4F(1.0f));

	gaussian_filter(s1_sq, a_sq, 11, 1.5f * 1.5f);
	add_weighted_image(s1_sq, vec4F(1), mu1_sq, vec4F(-1), vec4F(0), s1_sq);

	gaussian_filter(s2_sq, b_sq, 11, 1.5f * 1.5f);
	add_weighted_image(s2_sq, vec4F(1), mu2_sq, vec4F(-1), vec4F(0), s2_sq);

	gaussian_filter(s12, axb, 11, 1.5f * 1.5f);
	add_weighted_image(s12, vec4F(1), mu1_mu2, vec4F(-1), vec4F(0), s12);

	scale_image(mu1_mu2, t1, vec4F(2), vec4F(0));
	adds_image(t1, vec4F(C1), t1);

	scale_image(s12, t2, vec4F(2), vec4F(0));
	adds_image(t2, vec4F(C2), t2);

	mul_image(t1, t2, t3, vec4F(1));

	add_image(mu1_sq, mu2_sq, t1);
	adds_image(t1, vec4F(C1), t1);

	add_image(s1_sq, s2_sq, t2);
	adds_image(t2, vec4F(C2), t2);

	mul_image(t1, t2, t1, vec4F(1));

	div_image(t3, t1, smap, vec4F(1));

	return avg_image(smap);
}

vec4F compute_ssim(const image_u8& a, const image_u8& b, bool luma)
{
	image_u8 ta(a), tb(b);

	if ((ta.width() != tb.width()) || (ta.height() != tb.height()))
	{
		fprintf(stderr, "compute_ssim: Cropping input images to equal dimensions\n");

		const uint32_t w = std::min(a.width(), b.width());
		const uint32_t h = std::min(a.height(), b.height());
		ta.crop(w, h);
		tb.crop(w, h);
	}

	if (!ta.width() || !ta.height())
	{
		assert(0);
		return vec4F(0);
	}

	if (luma)
	{
		for (uint32_t y = 0; y < ta.height(); y++)
		{
			for (uint32_t x = 0; x < ta.width(); x++)
			{
				ta(x, y).set((uint8_t)ta(x, y).get_luma(), ta(x, y).a);
				tb(x, y).set((uint8_t)tb(x, y).get_luma(), tb(x, y).a);
			}
		}
	}

	imagef fta, ftb;

	fta.set(ta);
	ftb.set(tb);

	return compute_ssim(fta, ftb);
}

bool save_dds(const char* pFilename, uint32_t width, uint32_t height, const void* pBlocks, uint32_t pixel_format_bpp, DXGI_FORMAT dxgi_format, bool srgb, bool force_dx10_header)
{
	(void)srgb;

	FILE* pFile = NULL;
#ifdef _MSC_VER
	fopen_s(&pFile, pFilename, "wb");
#else
	pFile = fopen(pFilename, "wb");
#endif
	if (!pFile)
	{
		fprintf(stderr, "Failed creating file %s!\n", pFilename);
		return false;
	}

	fwrite("DDS ", 4, 1, pFile);

	DDSURFACEDESC2 desc;
	memset(&desc, 0, sizeof(desc));

	desc.dwSize = sizeof(desc);
	desc.dwFlags = DDSD_WIDTH | DDSD_HEIGHT | DDSD_PIXELFORMAT | DDSD_CAPS;

	desc.dwWidth = width;
	desc.dwHeight = height;

	desc.ddsCaps.dwCaps = DDSCAPS_TEXTURE;
	desc.ddpfPixelFormat.dwSize = sizeof(desc.ddpfPixelFormat);

	desc.ddpfPixelFormat.dwFlags |= DDPF_FOURCC;

	desc.lPitch = (((desc.dwWidth + 3) & ~3) * ((desc.dwHeight + 3) & ~3) * pixel_format_bpp) >> 3;
	desc.dwFlags |= DDSD_LINEARSIZE;

	desc.ddpfPixelFormat.dwRGBBitCount = 0;

	if ((!force_dx10_header) &&
		((dxgi_format == DXGI_FORMAT_BC1_UNORM) ||
			(dxgi_format == DXGI_FORMAT_BC3_UNORM) ||
			(dxgi_format == DXGI_FORMAT_BC4_UNORM) ||
			(dxgi_format == DXGI_FORMAT_BC5_UNORM)))
	{
		if (dxgi_format == DXGI_FORMAT_BC1_UNORM)
			desc.ddpfPixelFormat.dwFourCC = (uint32_t)PIXEL_FMT_FOURCC('D', 'X', 'T', '1');
		else if (dxgi_format == DXGI_FORMAT_BC3_UNORM)
			desc.ddpfPixelFormat.dwFourCC = (uint32_t)PIXEL_FMT_FOURCC('D', 'X', 'T', '5');
		else if (dxgi_format == DXGI_FORMAT_BC4_UNORM)
			desc.ddpfPixelFormat.dwFourCC = (uint32_t)PIXEL_FMT_FOURCC('A', 'T', 'I', '1');
		else if (dxgi_format == DXGI_FORMAT_BC5_UNORM)
			desc.ddpfPixelFormat.dwFourCC = (uint32_t)PIXEL_FMT_FOURCC('A', 'T', 'I', '2');

		fwrite(&desc, sizeof(desc), 1, pFile);
	}
	else
	{
		desc.ddpfPixelFormat.dwFourCC = (uint32_t)PIXEL_FMT_FOURCC('D', 'X', '1', '0');

		fwrite(&desc, sizeof(desc), 1, pFile);

		DDS_HEADER_DXT10 hdr10;
		memset(&hdr10, 0, sizeof(hdr10));

		// Not all tools support DXGI_FORMAT_BC7_UNORM_SRGB (like NVTT), but ddsview in DirectXTex pays attention to it. So not sure what to do here.
		// For best compatibility just write DXGI_FORMAT_BC7_UNORM.
		//hdr10.dxgiFormat = srgb ? DXGI_FORMAT_BC7_UNORM_SRGB : DXGI_FORMAT_BC7_UNORM;
		hdr10.dxgiFormat = dxgi_format; // DXGI_FORMAT_BC7_UNORM;
		hdr10.resourceDimension = D3D10_RESOURCE_DIMENSION_TEXTURE2D;
		hdr10.arraySize = 1;

		fwrite(&hdr10, sizeof(hdr10), 1, pFile);
	}

	fwrite(pBlocks, desc.lPitch, 1, pFile);

	if (fclose(pFile) == EOF)
	{
		fprintf(stderr, "Failed writing to DDS file %s!\n", pFilename);
		return false;
	}

	return true;
}

void strip_extension(std::string& s)
{
	for (int32_t i = (int32_t)s.size() - 1; i >= 0; i--)
	{
		if (s[i] == '.')
		{
			s.resize(i);
			break;
		}
	}
}

void strip_path(std::string& s)
{
	for (int32_t i = (int32_t)s.size() - 1; i >= 0; i--)
	{
		if ((s[i] == '/') || (s[i] == ':') || (s[i] == '\\'))
		{
			s.erase(0, i + 1);
			break;
		}
	}
}

uint32_t hash_hsieh(const uint8_t* pBuf, size_t len)
{
	if (!pBuf || !len)
		return 0;

	uint32_t h = static_cast<uint32_t>(len);

	const uint32_t bytes_left = len & 3;
	len >>= 2;

	while (len--)
	{
		const uint16_t* pWords = reinterpret_cast<const uint16_t*>(pBuf);

		h += pWords[0];

		const uint32_t t = (pWords[1] << 11) ^ h;
		h = (h << 16) ^ t;

		pBuf += sizeof(uint32_t);

		h += h >> 11;
	}

	switch (bytes_left)
	{
	case 1:
		h += *reinterpret_cast<const signed char*>(pBuf);
		h ^= h << 10;
		h += h >> 1;
		break;
	case 2:
		h += *reinterpret_cast<const uint16_t*>(pBuf);
		h ^= h << 11;
		h += h >> 17;
		break;
	case 3:
		h += *reinterpret_cast<const uint16_t*>(pBuf);
		h ^= h << 16;
		h ^= (static_cast<signed char>(pBuf[sizeof(uint16_t)])) << 18;
		h += h >> 11;
		break;
	default:
		break;
	}

	h ^= h << 3;
	h += h >> 5;
	h ^= h << 4;
	h += h >> 17;
	h ^= h << 25;
	h += h >> 6;

	return h;
}

float compute_block_max_std_dev(const color_quad_u8* pPixels, uint32_t block_width, uint32_t block_height, uint32_t num_comps)
{
	tracked_stat comp_stats[4];

	for (uint32_t y = 0; y < block_height; y++)
	{
		for (uint32_t x = 0; x < block_width; x++)
		{
			const color_quad_u8* pPixel = pPixels + x + y * block_width;

			for (uint32_t c = 0; c < num_comps; c++)
				comp_stats[c].update(pPixel->m_c[c]);
		}
	}

	float max_std_dev = 0.0f;
	for (uint32_t i = 0; i < num_comps; i++)
		max_std_dev = std::max(max_std_dev, comp_stats[i].get_std_dev());
	return max_std_dev;
}

const uint32_t ASTC_SIG = 0x5CA1AB13;

#pragma pack(push, 1)
struct astc_header
{
	uint32_t m_sig;
	uint8_t m_block_x;
	uint8_t m_block_y;
	uint8_t m_block_z;
	uint8_t m_width[3];
	uint8_t m_height[3];
	uint8_t m_depth[3];
};
#pragma pack(pop)

bool save_astc_file(const char* pFilename, block16_vec& blocks, uint32_t width, uint32_t height, uint32_t block_width, uint32_t block_height)
{
	FILE* pFile = nullptr;

#ifdef _MSC_VER	
	fopen_s(&pFile, pFilename, "wb");
#else
	pFile = fopen(pFilename, "wb");
#endif

	if (!pFile)
		return false;

	astc_header hdr;
	memset(&hdr, 0, sizeof(hdr));

	hdr.m_sig = ASTC_SIG;
	hdr.m_block_x = (uint8_t)block_width;
	hdr.m_block_y = (uint8_t)block_height;
	hdr.m_block_z = 1;
	hdr.m_width[0] = (uint8_t)(width);
	hdr.m_width[1] = (uint8_t)(width >> 8);
	hdr.m_width[2] = (uint8_t)(width >> 16);
	hdr.m_height[0] = (uint8_t)(height);
	hdr.m_height[1] = (uint8_t)(height >> 8);
	hdr.m_height[2] = (uint8_t)(height >> 16);
	hdr.m_depth[0] = 1;
	fwrite(&hdr, sizeof(hdr), 1, pFile);

	fwrite(blocks.data(), 16, blocks.size(), pFile);
	if (fclose(pFile) == EOF)
		return false;

	return true;
}

bool load_astc_file(const char* pFilename, block16_vec& blocks, uint32_t& width, uint32_t& height, uint32_t& block_width, uint32_t& block_height)
{
	FILE* pFile = nullptr;

#ifdef _MSC_VER
	fopen_s(&pFile, pFilename, "rb");
#else
	pFile = fopen(pFilename, "rb");
#endif

	if (!pFile)
		return false;

	astc_header hdr;
	if (fread(&hdr, sizeof(hdr), 1, pFile) != 1)
	{
		fclose(pFile);
		return false;
	}

	if (hdr.m_sig != ASTC_SIG)
	{
		fclose(pFile);
		return false;
	}

	width = hdr.m_width[0] + (hdr.m_width[1] << 8) + (hdr.m_width[2] << 16);
	height = hdr.m_height[0] + (hdr.m_height[1] << 8) + (hdr.m_height[2] << 16);
	uint32_t depth = hdr.m_depth[0] + (hdr.m_depth[1] << 8) + (hdr.m_depth[2] << 16);

	if ((width < 1) || (width > 32768) || (height < 1) || (height > 32768))
		return false;
	if ((hdr.m_block_z != 1) || (depth != 1))
		return false;

	block_width = hdr.m_block_x;
	block_height = hdr.m_block_y;

	if ((block_width < 4) || (block_width > 12) || (block_height < 4) || (block_height > 12))
		return false;

	uint32_t blocks_x = (width + block_width - 1) / block_width;
	uint32_t blocks_y = (height + block_height - 1) / block_height;
	uint32_t total_blocks = blocks_x * blocks_y;

	blocks.resize(total_blocks);

	if (fread(blocks.data(), 16, total_blocks, pFile) != total_blocks)
	{
		fclose(pFile);
		return false;
	}

	fclose(pFile);
	return true;
}

uint32_t get_deflate_size(const void* pData, size_t data_size)
{
	size_t comp_size = 0;
	void* pPre_RDO_Comp_data = tdefl_compress_mem_to_heap(pData, data_size, &comp_size, TDEFL_MAX_PROBES_MASK);// TDEFL_DEFAULT_MAX_PROBES);
	mz_free(pPre_RDO_Comp_data);

	if (comp_size > UINT32_MAX)
		return UINT32_MAX;

	return (uint32_t)comp_size;
}

} // namespace utils
