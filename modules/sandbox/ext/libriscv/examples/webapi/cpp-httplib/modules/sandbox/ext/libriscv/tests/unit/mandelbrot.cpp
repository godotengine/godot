#include <array>
#include <cmath>

#ifdef PNG_8BIT
using png_array_type = uint8_t;
inline void encode_color(png_array_type& px, int count, int max_count)
{
	px = count & 15;
}
#else
using png_array_type = uint32_t;
inline void encode_color(png_array_type& px, int count, int max_count)
{
	px = color_mapping[count & 15];
}
#endif

// Function to draw mandelbrot set
template <int DimX, int DimY, int MaxCount>
std::array<png_array_type, DimX * DimY>
fractal(double left, double top, double xside, double yside)
{
	std::array<png_array_type, DimX * DimY> bitmap {};

    // setting up the xscale and yscale
    const double xscale = xside / DimX;
    const double yscale = yside / DimY;

    // scanning every point in that rectangular area.
    // Each point represents a Complex number (x + yi).
    // Iterate that complex number
    for (int y = 0; y < DimY / 2; y++)
    for (int x = 0; x < DimX; x++)
    {
        double c_real = x * xscale + left;
        double c_imag = y * yscale + top;
        double z_real = 0;
        double z_imag = 0;
        int count = 0;

        // Calculate whether c(c_real + c_imag) belongs
        // to the Mandelbrot set or not and draw a pixel
        // at coordinates (x, y) accordingly
        // If you reach the Maximum number of iterations
        // and If the distance from the origin is
        // greater than 2 exit the loop
		#pragma GCC unroll 4
        while ((z_real * z_real + z_imag * z_imag < 4)
			&& (count < MaxCount))
        {
            // Calculate Mandelbrot function
            // z = z*z + c where z is a complex number
			double tempx =
            	z_real * z_real - z_imag * z_imag + c_real;
            z_imag = 2 * z_real * z_imag + c_imag;
            z_real = tempx;
            count++;
        }

		encode_color(bitmap[x + y * DimX], count, MaxCount);
    }
	for (int y = 0; y < DimY / 2; y++) {
		memcpy(&bitmap[(DimY-1 - y) * DimX], &bitmap[y * DimX], DimX * sizeof(png_array_type));
	}
	return bitmap;
}
