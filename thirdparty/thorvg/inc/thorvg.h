#ifndef _THORVG_H_
#define _THORVG_H_

#include <cstdint>
#include <functional>
#include <list>
#include <cstdarg>

#define TVG_VERSION_MAJOR 1  // for compile-time checks
#define TVG_VERSION_MINOR 0  // for compile-time checks
#define TVG_VERSION_MICRO 1  // for compile-time checks

#ifdef TVG_API
    #undef TVG_API
#endif

#ifndef TVG_STATIC
    #ifdef _WIN32
        #if TVG_BUILD
            #define TVG_API __declspec(dllexport)
        #else
            #define TVG_API __declspec(dllimport)
        #endif
    #elif (defined(__SUNPRO_C)  || defined(__SUNPRO_CC))
        #define TVG_API __global
    #else
        #if (defined(__GNUC__) && __GNUC__ >= 4) || defined(__INTEL_COMPILER)
            #define TVG_API __attribute__ ((visibility("default")))
        #else
            #define TVG_API
        #endif
    #endif
#else
    #define TVG_API
#endif

#ifdef TVG_DEPRECATED
    #undef TVG_DEPRECATED
#endif

#ifdef _WIN32
    #define TVG_DEPRECATED __declspec(deprecated)
#elif __GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1)
    #define TVG_DEPRECATED __attribute__ ((__deprecated__))
#else
    #define TVG_DEPRECATED
#endif

#define _TVG_DECLARE_PRIVATE(A) \
protected: \
    A(const A&) = delete; \
    const A& operator=(const A&) = delete; \
    A()

#define _TVG_DECLARE_PRIVATE_BASE(A) \
    _TVG_DECLARE_PRIVATE(A); \
public: \
    struct Impl; \
    Impl* pImpl

#define _TVG_DECLARE_PRIVATE_DERIVE(A) \
    _TVG_DECLARE_PRIVATE(A); \
protected: \
    ~A() {}

#define _TVG_DISABLE_CTOR(A) \
    A() = delete; \
    ~A() = delete

#define _TVG_DECLARE_ACCESSOR(A) \
    friend A

namespace tvg
{

struct RenderMethod;
struct Animation;
struct Shape;

/**
 * @defgroup ThorVG ThorVG
 * @brief ThorVG classes and enumerations providing C++ APIs.
 */

/**@{*/

/**
 * @brief Enumeration specifying the result from the APIs.
 *
 * All ThorVG APIs could potentially return one of the values in the list.
 * Please note that some APIs may additionally specify the reasons that trigger their return values.
 *
 */
enum struct Result
{
    Success = 0,           ///< The value returned in case of a correct request execution.
    InvalidArguments,      ///< The value returned in the event of a problem with the arguments given to the API - e.g. empty paths or null pointers.
    InsufficientCondition, ///< The value returned in case the request cannot be processed - e.g. asking for properties of an object, which does not exist.
    FailedAllocation,      ///< The value returned in case of unsuccessful memory allocation.
    MemoryCorruption,      ///< The value returned in the event of bad memory handling - e.g. failing in pointer releasing or casting
    NonSupport,            ///< The value returned in case of choosing unsupported engine features(options).
    Unknown = 255          ///< The value returned in all other cases.
};


/**
 * @brief Enumeration specifying the methods of combining the 8-bit color channels into 32-bit color.
 */
enum struct ColorSpace : uint8_t
{
    ABGR8888 = 0,      ///< The channels are joined in the order: alpha, blue, green, red. Colors are alpha-premultiplied.
    ARGB8888,          ///< The channels are joined in the order: alpha, red, green, blue. Colors are alpha-premultiplied.
    ABGR8888S,         ///< The channels are joined in the order: alpha, blue, green, red. Colors are un-alpha-premultiplied. @since 0.12
    ARGB8888S,         ///< The channels are joined in the order: alpha, red, green, blue. Colors are un-alpha-premultiplied. @since 0.12
    Grayscale8,        ///< One single channel data.
    Unknown = 255      ///< Unknown channel data. This is reserved for an initial ColorSpace value. @since 1.0
};


/**
 * @brief Enumeration to specify rendering engine behavior.
 *
 * @note The availability or behavior of @c SmartRender may vary depending on platform or backend support.
 *       It attempts to optimize rendering performance by updating only the regions  of the canvas that have
 *       changed between frames (partial redraw). This can be highly effective in scenarios  where most of the
 *       canvas remains static and only small portions are updated—such as simple animations or GUI interactions.
 *       However, in complex scenes where a large portion of the canvas changes frequently (e.g., full-screen animations
 *       or heavy object movements), the overhead of tracking changes and managing update regions may outweigh the benefits,
 *       resulting in decreased performance compared to the default rendering mode. Thus, it is recommended to benchmark
 *       both modes in your specific use case to determine the optimal setting.
 *
 * @since 1.0
 */
enum struct EngineOption : uint8_t
{
    None = 0,                   /**< No engine options are enabled. This may be used to explicitly disable all optional behaviors. */
    Default = 1 << 0,           /**< Uses the default rendering mode. */
    SmartRender = 1 << 1        /**< Enables automatic partial (smart) rendering optimizations. */
};


/**
 * @brief Enumeration specifying the values of the path commands accepted by ThorVG.
 */
enum struct PathCommand : uint8_t
{
    Close = 0, ///< Ends the current sub-path and connects it with its initial point. This command doesn't expect any points.
    MoveTo,    ///< Sets a new initial point of the sub-path and a new current point. This command expects 1 point: the starting position.
    LineTo,    ///< Draws a line from the current point to the given point and sets a new value of the current point. This command expects 1 point: the end-position of the line.
    CubicTo    ///< Draws a cubic Bezier curve from the current point to the given point using two given control points and sets a new value of the current point. This command expects 3 points: the 1st control-point, the 2nd control-point, the end-point of the curve.
};


/**
 * @brief Enumeration determining the ending type of a stroke in the open sub-paths.
 */
enum struct StrokeCap : uint8_t
{
    Butt = 0, ///< The stroke ends exactly at each of the two end-points of a sub-path. For zero length sub-paths no stroke is rendered.
    Round,    ///< The stroke is extended in both end-points of a sub-path by a half circle, with a radius equal to the half of a stroke width. For zero length sub-paths a full circle is rendered.
    Square    ///< The stroke is extended in both end-points of a sub-path by a rectangle, with the width equal to the stroke width and the length equal to the half of the stroke width. For zero length sub-paths the square is rendered with the size of the stroke width.
};


/**
 * @brief Enumeration determining the style used at the corners of joined stroked path segments.
 */
enum struct StrokeJoin : uint8_t
{
    Miter = 0, ///< The outer corner of the joined path segments is spiked. The spike is created by extension beyond the join point of the outer edges of the stroke until they intersect. In case the extension goes beyond the limit, the join style is converted to the Bevel style.
    Round,     ///< The outer corner of the joined path segments is rounded. The circular region is centered at the join point.
    Bevel      ///< The outer corner of the joined path segments is bevelled at the join point. The triangular region of the corner is enclosed by a straight line between the outer corners of each stroke.
};


/**
 * @brief Enumeration specifying how to fill the area outside the gradient bounds.
 */
enum struct FillSpread : uint8_t
{
    Pad = 0, ///< The remaining area is filled with the closest stop color.
    Reflect, ///< The gradient pattern is reflected outside the gradient area until the expected region is filled.
    Repeat   ///< The gradient pattern is repeated continuously beyond the gradient area until the expected region is filled.
};


/**
 * @brief Enumeration specifying the algorithm used to establish which parts of the shape are treated as the inside of the shape.
 */
enum struct FillRule : uint8_t
{
    NonZero = 0, ///< A line from the point to a location outside the shape is drawn. The intersections of the line with the path segment of the shape are counted. Starting from zero, if the path segment of the shape crosses the line clockwise, one is added, otherwise one is subtracted. If the resulting sum is non zero, the point is inside the shape.
    EvenOdd      ///< A line from the point to a location outside the shape is drawn and its intersections with the path segments of the shape are counted. If the number of intersections is an odd number, the point is inside the shape.
};


/**
 * @brief Enumeration indicating the method used in the mask of two objects - the target and the source.
 *
 * Notation: S(Source), T(Target), SA(Source Alpha), TA(Target Alpha)
 *
 * @see Paint::mask()
 */
enum struct MaskMethod : uint8_t
{
    None = 0,       ///< No Masking is applied.
    Alpha,          ///< Alpha Masking using the masking target's pixels as an alpha value.
    InvAlpha,       ///< Alpha Masking using the complement to the masking target's pixels as an alpha value.
    Luma,           ///< Alpha Masking using the grayscale (0.2126R + 0.7152G + 0.0722*B) of the masking target's pixels. @since 0.9
    InvLuma,        ///< Alpha Masking using the grayscale (0.2126R + 0.7152G + 0.0722*B) of the complement to the masking target's pixels. @since 0.11
    Add,            ///< Combines the target and source objects pixels using target alpha. (T * TA) + (S * (255 - TA)) @since 1.0
    Subtract,       ///< Subtracts the source color from the target color while considering their respective target alpha. (T * TA) - (S * (255 - TA)) @since 1.0
    Intersect,      ///< Computes the result by taking the minimum value between the target alpha and the source alpha and multiplies it with the target color. (T * min(TA, SA)) @since 1.0
    Difference,     ///< Calculates the absolute difference between the target color and the source color multiplied by the complement of the target alpha. abs(T - S * (255 - TA)) @since 1.0
    Lighten,        ///< Where multiple masks intersect, the highest transparency value is used. @since 1.0
    Darken          ///< Where multiple masks intersect, the lowest transparency value is used. @since 1.0
};


/**
 * @brief Enumeration indicates the method used for blending paint. Please refer to the respective formulas for each method.
 *
 * Notation: S(source paint as the top layer), D(destination as the bottom layer), Sa(source paint alpha), Da(destination alpha)
 *
 * @see Paint::blend()
 *
 * @since 0.15
 */
enum struct BlendMethod : uint8_t
{
    Normal = 0,        ///< Perform the alpha blending(default). S if (Sa == 255), otherwise (Sa * S) + (255 - Sa) * D
    Multiply,          ///< Takes the RGB channel values from 0 to 255 of each pixel in the top layer and multiples them with the values for the corresponding pixel from the bottom layer. (S * D)
    Screen,            ///< The values of the pixels in the two layers are inverted, multiplied, and then inverted again. (S + D) - (S * D)
    Overlay,           ///< Combines Multiply and Screen blend modes. (2 * S * D) if (D < 128), otherwise 255 - 2 * (255 - S) * (255 - D)
    Darken,            ///< Creates a pixel that retains the smallest components of the top and bottom layer pixels. min(S, D)
    Lighten,           ///< Only has the opposite action of Darken Only. max(S, D)
    ColorDodge,        ///< Divides the bottom layer by the inverted top layer. D / (255 - S)
    ColorBurn,         ///< Divides the inverted bottom layer by the top layer, and then inverts the result. 255 - (255 - D) / S
    HardLight,         ///< The same as Overlay but with the color roles reversed. (2 * S * D) if (S < 128), otherwise 255 - 2 * (255 - S) * (255 - D)
    SoftLight,         ///< The same as Overlay but with applying pure black or white does not result in pure black or white. (255 - 2 * S) * (D * D) + (2 * S * D)
    Difference,        ///< Subtracts the bottom layer from the top layer or the other way around, to always get a non-negative value. (S - D) if (S > D), otherwise (D - S)
    Exclusion,         ///< The result is twice the product of the top and bottom layers, subtracted from their sum. S + D - (2 * S * D)
    Hue,               ///< Combine with HSL(Sh + Ds + Dl) then convert it to RGB. @since 1.0
    Saturation,        ///< Combine with HSL(Dh + Ss + Dl) then convert it to RGB. @since 1.0
    Color,             ///< Combine with HSL(Sh + Ss + Dl) then convert it to RGB. @since 1.0
    Luminosity,        ///< Combine with HSL(Dh + Ds + Sl) then convert it to RGB. @since 1.0
    Add,               ///< Simply adds pixel values of one layer with the other. (S + D)
    Composition = 255  ///< For intermediate composition layers; suitable for use with Scene or Picture. @since 1.0
};


/**
 * @brief Enumeration that defines methods used for Scene Effects.
 *
 * This enum provides options to apply various post-processing effects to a scene.
 * Scene effects are typically applied to modify the final appearance of a rendered scene, such as blurring.
 *
 * @see Scene::add(SceneEffect effect, ...)
 *
 * @since 1.0
 */
enum struct SceneEffect : uint8_t
{
    Clear = 0,         ///< Clear all previously applied scene effects, restoring the scene to its original state.
    GaussianBlur,      ///< Apply a blur effect with a Gaussian filter. Param(4) = {sigma(double)[> 0], direction(int)[both: 0 / horizontal: 1 / vertical: 2], border(int)[duplicate: 0 / wrap: 1], quality(int)[0 - 100]}
    DropShadow,        ///< Apply a drop shadow effect with a Gaussian Blur filter. Param(8) = {color_R(int)[0 - 255], color_G(int)[0 - 255], color_B(int)[0 - 255], opacity(int)[0 - 255], angle(double)[0 - 360], distance(double), blur_sigma(double)[> 0], quality(int)[0 - 100]}
    Fill,              ///< Override the scene content color with a given fill information. Param(4) = {color_R(int)[0 - 255], color_G(int)[0 - 255], color_B(int)[0 - 255], opacity(int)[0 - 255]}
    Tint,              ///< Tinting the current scene color with a given black, white color parameters. Param(7) = {black_R(int)[0 - 255], black_G(int)[0 - 255], black_B(int)[0 - 255], white_R(int)[0 - 255], white_G(int)[0 - 255], white_B(int)[0 - 255], intensity(double)[0 - 100]}
    Tritone            ///< Apply a tritone color effect to the scene using three color parameters for shadows, midtones, and highlights. A blending factor determines the mix between the original color and the tritone colors. Param(9) = {Shadow_R(int)[0 - 255], Shadow_G(int)[0 - 255], Shadow_B(int)[0 - 255], Midtone_R(int)[0 - 255], Midtone_G(int)[0 - 255], Midtone_B(int)[0 - 255], Highlight_R(int)[0 - 255], Highlight_G(int)[0 - 255], Highlight_B(int)[0 - 255], Blend(int)[0 - 255]}
};


/**
 * @brief Enumeration that defines methods used for wrapping text.
 *
 * This enum provides options to control how text is wrapped when it exceeds the available space.
 * Wrapping affects the layout and flow of text in the rendering area.
 *
 * @see Text::wrap(TextWrap mode)
 *
 * @since 1.0
 */
enum struct TextWrap : uint8_t
{
    None = 0,      ///< Do not wrap text. Text is rendered on a single line and may overflow the bounding area.
    Character,     ///< Wrap at the character level. If a word cannot fit, it is broken into individual characters to fit the line.
    Word,          ///< Wrap at the word level. Words that do not fit are moved to the next line.
    Smart,         ///< Smart choose wrapping method: word wrap first, falling back to character wrap if a word does not fit.
    Ellipsis       ///< Truncate overflowing text and append an ellipsis ("...") at the end. Typically used for single-line labels.
};


/**
 * @brief Enumeration specifying the ThorVG class type value.
 *
 * ThorVG's drawing objects can return class type values, allowing you to identify the specific class of each object.
 *
 * @see Paint::type()
 * @see Fill::type()
 *
 * @since 1.0
 */
enum struct Type : uint8_t
{
    Undefined = 0,         ///< Unknown class
    Shape,                 ///< Shape class
    Scene,                 ///< Scene class
    Picture,               ///< Picture class
    Text,                  ///< Text class
    LinearGradient = 10,   ///< LinearGradient class
    RadialGradient         ///< RadialGradient class
};


/**
 * @brief A data structure representing a point in two-dimensional space.
 *
 * This structure defines a single point using Cartesian coordinates.
 * It is typically used for specifying positions or coordinates in 2D graphics.
 */
struct Point
{
    float x;  ///< The x-coordinate of the point.
    float y;  ///< The y-coordinate of the point.
};


/**
 * @brief A data structure representing a three-dimensional matrix.
 *
 * The elements e11, e12, e21 and e22 represent the rotation matrix, including the scaling factor.
 * The elements e13 and e23 determine the translation of the object along the x and y-axis, respectively.
 * The elements e31 and e32 are set to 0, e33 is set to 1.
 */
struct Matrix
{
    float e11, e12, e13;
    float e21, e22, e23;
    float e31, e32, e33;
};


/**
 * @brief Describes the font metrics of a text object.
 *
 * Provides the basic vertical layout metrics used for text rendering,
 * such as ascent, descent, and line spacing (linegap).
 *
 * @see Text::metrics()
 * @note Experimental API
 */
struct TextMetrics
{
    float ascent;   ///< Distance from the baseline to the top of the highest glyph (usually positive).
    float descent;  ///< Distance from the baseline to the bottom of the lowest glyph (usually negative, as in TTF).
    float linegap;  ///< Additional spacing recommended between lines (leading).
    float advance;  ///< The total vertical advance between lines of text: ascent - descent + linegap (i.e., ascent + |descent| + linegap when descent is negative).
};


/**
 * @class Paint
 *
 * @brief An abstract class for managing graphical elements.
 *
 * A graphical element in TVG is any object composed into a Canvas.
 * Paint represents such a graphical object and its behaviors such as duplication, transformation and composition.
 * TVG recommends the user to regard a paint as a set of volatile commands. They can prepare a Paint and then request a Canvas to run them.
 */
struct TVG_API Paint
{
    /**
     * @brief Retrieves the parent paint object.
     *
     * This function returns a pointer to the parent object if the current paint
     * belongs to one. Otherwise, it returns @c nullptr.
     *
     * @return A pointer to the parent object if available, otherwise @c nullptr.
     *
     * @see Scene::add()
     * @see Canvas::add()
     *
     * @since 1.0
    */
    const Paint* parent() const noexcept;

    /**
     * @brief Sets the visibility of the Paint object.
     *
     * This is useful for selectively excluding paint objects during rendering.
     *
     * @param[in] on A boolean flag indicating visibility. The default is @c true.
     *               @c true, the object will be rendered by the engine.
     *               @c false, the object will be excluded from the drawing process.
     *
     * @note An invisible object is not considered inactive—it may still participate
     *       in internal update processing if its properties are updated, but it will not
     *       be taken into account for the final drawing output. To completely deactivate
     *       a paint object, remove it from the canvas.
     *
     * @see Paint::visible() const
     * @see Result Canvas::remove(Paint* paint)
     *
     * @since 1.0
     */
    Result visible(bool on) noexcept;

    /**
     * @brief Sets the angle by which the object is rotated.
     *
     * The angle in measured clockwise from the horizontal axis.
     * The rotational axis passes through the point on the object with zero coordinates.
     *
     * @param[in] degree The value of the angle in degrees.
     *
     * @retval Result::InsufficientCondition in case a custom transform is applied.
     * @see Paint::transform()
     */
    Result rotate(float degree) noexcept;

    /**
     * @brief Sets the scale value of the object.
     *
     * @param[in] factor The value of the scaling factor. The default value is 1.
     *
     * @retval Result::InsufficientCondition in case a custom transform is applied.
     * @see Paint::transform()
     */
    Result scale(float factor) noexcept;

    /**
     * @brief Sets the values by which the object is moved in a two-dimensional space.
     *
     * The origin of the coordinate system is in the upper-left corner of the canvas.
     * The horizontal and vertical axes point to the right and down, respectively.
     *
     * @param[in] x The value of the horizontal shift.
     * @param[in] y The value of the vertical shift.
     *
     * @retval Result::InsufficientCondition in case a custom transform is applied.
     * @see Paint::transform()
     */
    Result translate(float x, float y) noexcept;

    /**
     * @brief Sets the matrix of the affine transformation for the object.
     *
     * The augmented matrix of the transformation is expected to be given.
     *
     * @param[in] m The 3x3 augmented matrix.
     */
    Result transform(const Matrix& m) noexcept;

    /**
     * @brief Gets the matrix of the affine transformation of the object.
     *
     * The values of the matrix can be set by the transform() API, as well by the translate(),
     * scale() and rotate(). In case no transformation was applied, the identity matrix is returned.
     *
     * @return The augmented transformation matrix.
     *
     * @since 0.4
     */
    Matrix& transform() noexcept;

    /**
     * @brief Sets the opacity of the object.
     *
     * @param[in] o The opacity value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque.
     *
     * @note Setting the opacity with this API may require multiple render pass for composition. It is recommended to avoid changing the opacity if possible.
     */
    Result opacity(uint8_t o) noexcept;

    /**
     * @brief Sets the masking target object and the masking method.
     *
     * @param[in] target The paint of the target object.
     * @param[in] method The method used to mask the source object with the target.
     *
     * @retval Result::InsufficientCondition if the target has already belonged to another paint.
     * @retval Result::InvalidArguments @p method equals @c MaskMethod::None and @p target is not @c nullptr.
     */
    Result mask(Paint* target, MaskMethod method) noexcept;

    /**
     * @brief Clip the drawing region of the paint object.
     *
     * This function restricts the drawing area of the paint object to the specified shape's paths.
     *
     * @param[in] clipper The shape object as the clipper.
     *
     * @retval Result::InsufficientCondition if the @p clipper has already belonged to another paint.
     *
     * @see Paint::clip()
     *
     * @since 1.0
     */
    Result clip(Shape* clipper) noexcept;

    /**
     * @brief Sets the blending method for the paint object.
     *
     * The blending feature allows you to combine colors to create visually appealing effects, including transparency, lighting, shading, and color mixing, among others.
     * its process involves the combination of colors or images from the source paint object with the destination (the lower layer image) using blending operations.
     * The blending operation is determined by the chosen @p BlendMethod, which specifies how the colors or images are combined.
     *
     * @param[in] method The blending method to be set.
     *
     * @since 1.0
     */
    Result blend(BlendMethod method) noexcept;

    /**
     * @brief Retrieves the object-oriented bounding box (OBB) of the paint object in canvas space.
     *
     * This function returns the bounding box of the paint, as an oriented bounding box (OBB) after transformations are applied.
     * The returned values @p pt4 may have invalid if the operation fails. Thus, please check the retval.
     *
     * This bounding box can be used to obtain the transformed bounding region in canvas space
     * by taking the geometry's axis-aligned bounding box (AABB) in the object's local coordinate space
     * and applying the object's transformations.
     *
     * @param[out] pt4 An array of four points representing the bounding box. The array size must be 4.
     *
     * @retval Result::InsufficientCondition If the paint has not been updated by the canvas.
     * 
     * @see Paint::bounds(float* x, float* y, float* w, float* h)
     * @see Canvas::update()
     *
     * @since 1.0
     */
    Result bounds(Point* pt4) noexcept;

    /**
     * @brief Retrieves the axis-aligned bounding box (AABB) of the paint object in canvas space.
     *
     * Returns the bounding box of the paint as an axis-aligned bounding box (AABB), with all relevant transformations applied.
     * The returned values @p x, @p y, @p w, @p h, may have invalid if the operation fails. Thus, please check the retval.
     *
     * This bounding box can be used to determine the actual rendered area of the object on the canvas,
     * for purposes such as hit-testing, culling, or layout calculations.
     *
     * @param[out] x The x-coordinate of the upper-left corner of the bounding box.
     * @param[out] y The y-coordinate of the upper-left corner of the bounding box.
     * @param[out] w The width of the bounding box.
     * @param[out] h The height of the bounding box.
     *
     * @retval Result::InsufficientCondition If the paint has not been updated by the canvas.
     *
     * @see Paint::bounds(Point* pt4)
     * @see Canvas::update()
     */
    Result bounds(float* x, float* y, float* w, float* h) noexcept;

    /**
     * @brief Checks whether a given region intersects the filled area of the paint.
     *
     * This function determines whether the specified rectangular region—defined by (`x`, `y`, `w`, `h`)—
     * intersects the geometric fill region of the paint object.
     *
     * This is useful for hit-testing purposes, such as detecting whether a user interaction (e.g., touch or click)
     * occurs within a painted region.
     *
     * The paint must be updated in a Canvas beforehand—typically after the Canvas has been
     * drawn and synchronized.
     *
     * @param[in] x The x-coordinate of the top-left corner of the test region.
     * @param[in] y The y-coordinate of the top-left corner of the test region.
     * @param[in] w The width of the region to test. Must be greater than 0; defaults to 1.
     * @param[in] h The height of the region to test. Must be greater than 0; defaults to 1.
     *
     * @return @c true if any part of the region intersects the filled area; otherwise, @c false.
     *
     * @note To test a single point, set the region size to w = 1, h = 1.
     * @note For efficiency, an AABB (axis-aligned bounding box) test is performed internally before precise hit detection.
     * @note This test does not take into account the results of blending or masking.
     * @note This test does take into account the the hidden paints as well. @see Paint::visible()
     * @since 1.0
     */
    bool intersects(int32_t x, int32_t y, int32_t w = 1, int32_t h = 1) noexcept;

    /**
     * @brief Duplicates the object.
     *
     * Creates a new object and sets its all properties as in the original object.
     *
     * @return The created object when succeed, @c nullptr otherwise.
     */
    Paint* duplicate() const noexcept;

    /**
     * @brief Gets the opacity value of the object.
     *
     * @return The opacity value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque.
     */
    uint8_t opacity() const noexcept;

    /**
     * @brief Gets the masking target object and the masking method.
     *
     * @param[out] target The paint of the target object.
     *
     * @return The method used to mask the source object with the target.
     *
     * @since 0.5
     */
    MaskMethod mask(const Paint** target) const noexcept;

    /**
     * @brief Get the clipper shape of the paint object.
     *
     * This function returns the clipper that has been previously set to this paint object.
     *
     * @return The shape object used as the clipper, or @c nullptr if no clipper is set.
     *
     * @see Paint::clip(Shape* clipper)
     *
     * @since 1.0
     */
    Shape* clip() const noexcept;

    /**
     * @brief Gets the current visibility status of the Paint object.
     *
     * @return true if the object is visible and will be rendered.
     *         false if the object is hidden and will not be rendered.
     *
     * @see Paint::visible(bool on)
     *
     * @since 1.0
     */
    bool visible() const noexcept;

    /**
     * @brief Increment the reference count for the Paint instance.
     *
     * This method increases the reference count of the Paint object, allowing shared ownership and control over its lifetime.
     *
     * @return The updated reference count after the increment by 1.
     *
     * @warning Please ensure that each call to ref() is paired with a corresponding call to unref() to prevent a dangling instance.
     *
     * @see Paint::unref()
     * @see Paint::refCnt()
     *
     * @since 1.0
     */
    uint16_t ref() noexcept;

    /**
     * @brief Decrement the reference count for the Paint instance.
     *
     * This method decreases the reference count of the Paint object by 1.
     * If the reference count reaches zero and the @p free flag is set to true, the Paint instance is automatically deleted.
     *
     * @param[in] free Flag indicating whether to delete the Paint instance when the reference count reaches zero.
     *
     * @return The updated reference count after the decrement.
     *
     * @see Paint::ref()
     * @see Paint::refCnt()
     *
     * @since 1.0
     */
    uint16_t unref(bool free = true) noexcept;

    /**
     * @brief Retrieve the current reference count of the Paint instance.
     *
     * This method provides the current reference count, allowing the user to check the shared ownership state of the Paint object.
     *
     * @return The current reference count of the Paint instance.
     *
     * @see Paint::ref()
     * @see Paint::unref()
     *
     * @since 1.0
     */
    uint16_t refCnt() const noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the Paint instance.
     *
     * @since 1.0
     */
    virtual Type type() const noexcept = 0;

    /**
     * @brief Unique ID of this instance.
     *
     * This is reserved to specify an paint instance in a scene.
     *
     * @since 1.0
     */
    uint32_t id = 0;

    /**
     * @brief Safely releases a Paint object.
     *
     * This is the counterpart to the `gen()` API, and releases the given Paint object safely, 
     * handling @c nullptr and managing ownership properly.
     *
     * @param[in] paint A Paint object to release.
     *
     * @since 1.0
     */
    static void rel(Paint* paint) noexcept;

protected:
    virtual ~Paint();

    _TVG_DECLARE_PRIVATE_BASE(Paint);
};


/**
 * @class Fill
 *
 * @brief An abstract class representing the gradient fill of the Shape object.
 *
 * It contains the information about the gradient colors and their arrangement
 * inside the gradient bounds. The gradients bounds are defined in the LinearGradient
 * or RadialGradient class, depending on the type of the gradient to be used.
 * It specifies the gradient behavior in case the area defined by the gradient bounds
 * is smaller than the area to be filled.
 */
struct TVG_API Fill
{
    /**
     * @brief A data structure storing the information about the color and its relative position inside the gradient bounds.
     */
    struct ColorStop
    {
        float offset; /**< The relative position of the color. */
        uint8_t r;    /**< The red color channel value in the range [0 ~ 255]. */
        uint8_t g;    /**< The green color channel value in the range [0 ~ 255]. */
        uint8_t b;    /**< The blue color channel value in the range [0 ~ 255]. */
        uint8_t a;    /**< The alpha channel value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque. */
    };

    virtual ~Fill();

    /**
     * @brief Sets the parameters of the colors of the gradient and their position.
     *
     * @param[in] colorStops An array of ColorStop data structure.
     * @param[in] cnt The count of the @p colorStops array equal to the colors number used in the gradient.
     */
    Result colorStops(const ColorStop* colorStops, uint32_t cnt) noexcept;

    /**
     * @brief Sets the FillSpread value, which specifies how to fill the area outside the gradient bounds.
     *
     * @param[in] s The FillSpread value.
     */
    Result spread(FillSpread s) noexcept;

    /**
     * @brief Sets the matrix of the affine transformation for the gradient fill.
     *
     * The augmented matrix of the transformation is expected to be given.
     *
     * @param[in] m The 3x3 augmented matrix.
     */
    Result transform(const Matrix& m) noexcept;

    /**
     * @brief Gets the parameters of the colors of the gradient, their position and number.
     *
     * @param[out] colorStops A pointer to the memory location, where the array of the gradient's ColorStop is stored.
     *
     * @return The number of colors used in the gradient. This value corresponds to the length of the @p colorStops array.
     */
    uint32_t colorStops(const ColorStop** colorStops) const noexcept;

    /**
     * @brief Gets the FillSpread value of the fill.
     *
     * @return The FillSpread value of this Fill.
     */
    FillSpread spread() const noexcept;

    /**
     * @brief Gets the matrix of the affine transformation of the gradient fill.
     *
     * In case no transformation was applied, the identity matrix is returned.
     *
     * @return The augmented transformation matrix.
     */
    Matrix& transform() const noexcept;

    /**
     * @brief Creates a copy of the Fill object.
     *
     * Return a newly created Fill object with the properties copied from the original.
     *
     * @return A copied Fill object when succeed, @c nullptr otherwise.
     */
    Fill* duplicate() const noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the Fill instance.
     *
     * @since 1.0
     */
    virtual Type type() const noexcept = 0;

    _TVG_DECLARE_PRIVATE_BASE(Fill);
};


/**
 * @class Canvas
 *
 * @brief An abstract class for drawing graphical elements.
 *
 * A canvas is an entity responsible for drawing the target. It sets up the drawing engine and the buffer, which can be drawn on the screen. It also manages given Paint objects.
 *
 * @note A Canvas behavior depends on the raster engine though the final content of the buffer is expected to be identical.
 * @warning The Paint objects belonging to one Canvas can't be shared among multiple Canvases.
 */
struct TVG_API Canvas
{
    virtual ~Canvas();

    /**
     * @brief Returns the list of paints currently held by the Canvas.
     *
     * This function provides a list of paint nodes, allowing users to access scene-graph information.
     *
     * @warning Please avoid accessing the paints during Canvas update/draw. You can access them after calling sync().
     * @see Canvas::add()
     * @see Canvas::remove()
     *
     * @warning This is read-only. Do not modify the list.
     * @note 1.0
     */
    const std::list<Paint*>& paints() const noexcept;

    /**
     * @brief Adds a paint object to the canvas root scene.
     *
     * Appends a paint object to the root scene of the canvas. If the optional @p at
     * parameter is provided, the paint object is inserted immediately before the
     * specified paint in the root scene. If @p at is @c nullptr, the paint object
     * is appended to the end of the root scene.
     *
     * @param[in] target A pointer to the Paint object to be added to the root scene.
     *                   This parameter must not be @c nullptr.
     * @param[in] at     A pointer to an existing Paint object in the root scene
     *                   before which @p target will be inserted. If @c nullptr,
     *                   @p target is appended to the end of the root scene.
     *                   The default value is @c nullptr.
     *
     * @note Ownership of the @p target object is transferred to the canvas upon
     *       successful addition. To retain ownership, call @ref Paint::ref()
     *       before adding it to the canvas.
     * @note The rendering order of paint objects follows their order in the root
     *       scene. If layering is required, ensure the paints are added in the
     *       desired order.
     *
     * @see Canvas::paints()
     * @see Canvas::remove()
     * @see Paint::ref()
     *
     * @since 1.0
     */
    Result add(Paint* target, Paint* at = nullptr) noexcept;

    /**
     * @brief Removes a paint object or all paint objects from the root scene.
     *
     * This function removes a specified paint object from the root scene. If no paint
     * object is specified (i.e., the default @c nullptr is used), the function
     * performs to clear all paints from the root scene.
     *
     * @param[in] paint A pointer to the Paint object to be removed from the root scene.
     *                  If @c nullptr, remove all the paints from the root scene.
     *
     * @see Canvas::add()
     * @see Canvas::paints()
     *
     * @since 1.0
     */
    Result remove(Paint* paint = nullptr) noexcept;

    /**
     * @brief Requests the canvas to update modified paint objects in preparation for rendering.
     *
     * This function triggers an internal update for all paint instances that have been modified
     * since the last update. It ensures that the canvas state is ready for accurate rendering.
     *
     * @retval Result::InsufficientCondition The canvas is not properly prepared.
     *         This may occur if the canvas target has not been set or if the update is called during drawing.
     *         Call Canvas::sync() before trying.
     *
     * @note Only paint objects that have been changed will be processed.
     * @note If the canvas is configured with multiple threads, the update may be performed asynchronously.
     *
     * @see Canvas::sync()
     */
    Result update() noexcept;

    /**
     * @brief Requests the canvas to render the Paint objects.
     *
     * @param[in] clear If @c true, clears the target buffer to zero before drawing.
     *
     * @retval Result::InsufficientCondition The canvas is not properly prepared.
     *         This may occur if Canvas::target() has not been set or if draw() is called multiple times
     *         without calling Canvas::sync() in between.
     *
     * @note Clearing the buffer is unnecessary if the canvas will be fully covered 
     *       with opaque content. Skipping the clear can improve performance.
     * @note Drawing may be performed asynchronously if the thread count is greater than zero.
     *       To ensure the drawing process is complete, call sync() afterwards.
     * @note If the canvas has not been updated prior to Canvas::draw(), it may implicitly perform Canvas::update().
     *
     * @see Canvas::sync()
     * @see Canvas::update()
     */
    Result draw(bool clear = false) noexcept;

    /**
     * @brief Sets the drawing region of the canvas.
     *
     * This function defines a rectangular area of the canvas to be used for drawing operations.
     * The specified viewport clips rendering output to the boundaries of that rectangle.
     *
     * Please note that changing the viewport is only allowed at the beginning of the rendering sequence—that is, after calling Canvas::sync().
     *
     * @param[in] x The x-coordinate of the upper-left corner of the rectangle.
     * @param[in] y The y-coordinate of the upper-left corner of the rectangle.
     * @param[in] w The width of the rectangle.
     * @param[in] h The height of the rectangle.
     *
     * @retval Result::InsufficientCondition If the canvas is not in a synced state.
     *
     * @see Canvas::sync()
     * @see SwCanvas::target()
     * @see GlCanvas::target()
     * @see WgCanvas::target()
     *
     * @warning Changing the viewport is not allowed after calling Canvas::add(),
     *          Canvas::remove(), Canvas::update(), or Canvas::draw().
     *
     * @note When the target is reset, the viewport will also be reset to match the target size.
     * @since 0.15
     */
    Result viewport(int32_t x, int32_t y, int32_t w, int32_t h) noexcept;

    /**
     * @brief Guarantees that drawing task is finished.
     *
     * The Canvas rendering can be performed asynchronously. To make sure that rendering is finished,
     * the sync() must be called after the draw() regardless of threading.
     *
     * @see Canvas::draw()
     */
    Result sync() noexcept;

    _TVG_DECLARE_PRIVATE_BASE(Canvas);
};


/**
 * @class LinearGradient
 *
 * @brief A class representing the linear gradient fill of the Shape object.
 *
 * Besides the APIs inherited from the Fill class, it enables setting and getting the linear gradient bounds.
 * The behavior outside the gradient bounds depends on the value specified in the spread API.
 *
 * @warning This class is not designed for inheritance.
 */
struct TVG_API LinearGradient : Fill
{
    /**
     * @brief Sets the linear gradient bounds.
     *
     * The bounds of the linear gradient are defined as a surface constrained by two parallel lines crossing
     * the given points (@p x1, @p y1) and (@p x2, @p y2), respectively. Both lines are perpendicular to the line linking
     * (@p x1, @p y1) and (@p x2, @p y2).
     *
     * @param[in] x1 The horizontal coordinate of the first point used to determine the gradient bounds.
     * @param[in] y1 The vertical coordinate of the first point used to determine the gradient bounds.
     * @param[in] x2 The horizontal coordinate of the second point used to determine the gradient bounds.
     * @param[in] y2 The vertical coordinate of the second point used to determine the gradient bounds.
     *
     * @note In case the first and the second points are equal, an object is filled with a single color using the last color specified in the colorStops().
     * @see Fill::colorStops()
     */
    Result linear(float x1, float y1, float x2, float y2) noexcept;

    /**
     * @brief Gets the linear gradient bounds.
     *
     * The bounds of the linear gradient are defined as a surface constrained by two parallel lines crossing
     * the given points (@p x1, @p y1) and (@p x2, @p y2), respectively. Both lines are perpendicular to the line linking
     * (@p x1, @p y1) and (@p x2, @p y2).
     *
     * @param[out] x1 The horizontal coordinate of the first point used to determine the gradient bounds.
     * @param[out] y1 The vertical coordinate of the first point used to determine the gradient bounds.
     * @param[out] x2 The horizontal coordinate of the second point used to determine the gradient bounds.
     * @param[out] y2 The vertical coordinate of the second point used to determine the gradient bounds.
     */
    Result linear(float* x1, float* y1, float* x2, float* y2) const noexcept;

    /**
     * @brief Creates a new LinearGradient object.
     *
     * @return A new LinearGradient object.
     */
    static LinearGradient* gen() noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the LinearGradient instance.
     *
     * @since 1.0
     */
    Type type() const noexcept override;

    _TVG_DECLARE_PRIVATE(LinearGradient);
};


/**
 * @class RadialGradient
 *
 * @brief A class representing the radial gradient fill of the Shape object.
 *
 * @warning This class is not designed for inheritance.
 */
struct TVG_API RadialGradient : Fill
{
    /**
     * @brief Sets the radial gradient attributes.
     *
     * The radial gradient is defined by the end circle with a center (@p cx, @p cy) and a radius @p r and
     * the start circle with a center/focal point (@p fx, @p fy) and a radius @p fr.
     * The gradient will be rendered such that the gradient stop at an offset of 100% aligns with the edge of the end circle
     * and the stop at an offset of 0% aligns with the edge of the start circle.
     *
     * @param[in] cx The horizontal coordinate of the center of the end circle.
     * @param[in] cy The vertical coordinate of the center of the end circle.
     * @param[in] r The radius of the end circle.
     * @param[in] fx The horizontal coordinate of the center of the start circle.
     * @param[in] fy The vertical coordinate of the center of the start circle.
     * @param[in] fr The radius of the start circle.
     *
     * @retval Result::InvalidArguments in case the radius @p r or @p fr value is negative.
     *
     * @note In case the radius @p r is zero, an object is filled with a single color using the last color specified in the colorStops().
     * @note In case the focal point (@p fx and @p fy) lies outside the end circle, it is projected onto the edge of the end circle.
     * @note If the start circle doesn't fully fit inside the end circle (after possible repositioning), the @p fr is reduced accordingly.
     * @note By manipulating the position and size of the focal point, a wide range of visual effects can be achieved, such as directing
     * the gradient focus towards a specific edge or enhancing the depth and complexity of shading patterns.
     * If a focal effect is not desired, simply align the focal point (@p fx and @p fy) with the center of the end circle (@p cx and @p cy)
     * and set the radius (@p fr) to zero. This will result in a uniform gradient without any focal variations.
     */
    Result radial(float cx, float cy, float r, float fx, float fy, float fr) noexcept;

    /**
     * @brief Gets the radial gradient attributes.
     *
     * @param[out] cx The horizontal coordinate of the center of the end circle.
     * @param[out] cy The vertical coordinate of the center of the end circle.
     * @param[out] r The radius of the end circle.
     * @param[out] fx The horizontal coordinate of the center of the start circle.
     * @param[out] fy The vertical coordinate of the center of the start circle.
     * @param[out] fr The radius of the start circle.
     *
     * @see RadialGradient::radial()
     */
    Result radial(float* cx, float* cy, float* r, float* fx = nullptr, float* fy = nullptr, float* fr = nullptr) const noexcept;

    /**
     * @brief Creates a new RadialGradient object.
     *
     * @return A new RadialGradient object.
     */
    static RadialGradient* gen() noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the LinearGradient instance.
     *
     * @since 1.0
     */
    Type type() const noexcept override;

    _TVG_DECLARE_PRIVATE(RadialGradient);
};


/**
 * @class Shape
 *
 * @brief A class representing two-dimensional figures and their properties.
 *
 * A shape has three major properties: shape outline, stroking, filling. The outline in the Shape is retained as the path.
 * Path can be composed by accumulating primitive commands such as moveTo(), lineTo(), cubicTo(), or complete shape interfaces such as appendRect(), appendCircle(), etc.
 * Path can consists of sub-paths. One sub-path is determined by a close command.
 *
 * The stroke of Shape is an optional property in case the Shape needs to be represented with/without the outline borders.
 * It's efficient since the shape path and the stroking path can be shared with each other. It's also convenient when controlling both in one context.
 *
 * @warning This class is not designed for inheritance.
 */
struct TVG_API Shape : Paint
{
    /**
     * @brief Resets the shape path.
     *
     * The transformation matrix, color, fill, and stroke properties are retained.
     *
     * @note The memory where the path data is stored is not deallocated at this stage to allow for caching.
     */
    Result reset() noexcept;

    /**
     * @brief Sets the initial point of the sub-path.
     *
     * The value of the current point is set to the given point.
     *
     * @param[in] x The horizontal coordinate of the initial point of the sub-path.
     * @param[in] y The vertical coordinate of the initial point of the sub-path.
     */
    Result moveTo(float x, float y) noexcept;

    /**
     * @brief Adds a new point to the sub-path, which results in drawing a line from the current point to the given end-point.
     *
     * The value of the current point is set to the given end-point.
     *
     * @param[in] x The horizontal coordinate of the end-point of the line.
     * @param[in] y The vertical coordinate of the end-point of the line.
     *
     * @note In case this is the first command in the path, it corresponds to the moveTo() call.
     */
    Result lineTo(float x, float y) noexcept;

    /**
     * @brief Adds new points to the sub-path, which results in drawing a cubic Bezier curve starting
     * at the current point and ending at the given end-point (@p x, @p y) using the control points (@p cx1, @p cy1) and (@p cx2, @p cy2).
     *
     * The value of the current point is set to the given end-point.
     *
     * @param[in] cx1 The horizontal coordinate of the 1st control point.
     * @param[in] cy1 The vertical coordinate of the 1st control point.
     * @param[in] cx2 The horizontal coordinate of the 2nd control point.
     * @param[in] cy2 The vertical coordinate of the 2nd control point.
     * @param[in] x The horizontal coordinate of the end-point of the curve.
     * @param[in] y The vertical coordinate of the end-point of the curve.
     *
     * @note In case this is the first command in the path, no data from the path are rendered.
     */
    Result cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y) noexcept;

    /**
     * @brief Closes the current sub-path by drawing a line from the current point to the initial point of the sub-path.
     *
     * The value of the current point is set to the initial point of the closed sub-path.
     *
     * @note In case the sub-path does not contain any points, this function has no effect.
     */
    Result close() noexcept;

    /**
     * @brief Appends a rectangle to the path.
     *
     * The rectangle with rounded corners can be achieved by setting non-zero values to @p rx and @p ry arguments.
     * The @p rx and @p ry values specify the radii of the ellipse defining the rounding of the corners.
     *
     * The position of the rectangle is specified by the coordinates of its upper-left corner - @p x and @p y arguments.
     *
     * The rectangle is treated as a new sub-path - it is not connected with the previous sub-path.
     *
     * The value of the current point is set to (@p x + @p rx, @p y) - in case @p rx is greater
     * than @p w/2 the current point is set to (@p x + @p w/2, @p y)
     *
     * @param[in] x The horizontal coordinate of the upper-left corner of the rectangle.
     * @param[in] y The vertical coordinate of the upper-left corner of the rectangle.
     * @param[in] w The width of the rectangle.
     * @param[in] h The height of the rectangle.
     * @param[in] rx The x-axis radius of the ellipse defining the rounded corners of the rectangle.
     * @param[in] ry The y-axis radius of the ellipse defining the rounded corners of the rectangle.
     * @param[in] cw Specifies the path direction: @c true for clockwise, @c false for counterclockwise.
     *
     * @note For @p rx and @p ry greater than or equal to the half of @p w and the half of @p h, respectively, the shape become an ellipse.
     */
    Result appendRect(float x, float y, float w, float h, float rx = 0, float ry = 0, bool cw = true) noexcept;

    /**
     * @brief Appends an ellipse to the path.
     *
     * The position of the ellipse is specified by the coordinates of its center - @p cx and @p cy arguments.
     *
     * The ellipse is treated as a new sub-path - it is not connected with the previous sub-path.
     *
     * The value of the current point is set to (@p cx, @p cy - @p ry).
     *
     * @param[in] cx The horizontal coordinate of the center of the ellipse.
     * @param[in] cy The vertical coordinate of the center of the ellipse.
     * @param[in] rx The x-axis radius of the ellipse.
     * @param[in] ry The y-axis radius of the ellipse.
     * @param[in] cw Specifies the path direction: @c true for clockwise, @c false for counterclockwise.
     *
     */
    Result appendCircle(float cx, float cy, float rx, float ry, bool cw = true) noexcept;

    /**
     * @brief Appends a given sub-path to the path.
     *
     * The current point value is set to the last point from the sub-path.
     * For each command from the @p cmds array, an appropriate number of points in @p pts array should be specified.
     * If the number of points in the @p pts array is different than the number required by the @p cmds array, the shape with this sub-path will not be displayed on the screen.
     *
     * @param[in] cmds The array of the commands in the sub-path.
     * @param[in] cmdCnt The number of the sub-path's commands.
     * @param[in] pts The array of the two-dimensional points.
     * @param[in] ptsCnt The number of the points in the @p pts array.
     *
     * @note The interface is designed for optimal path setting if the caller has a completed path commands already.
     */
    Result appendPath(const PathCommand* cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt) noexcept;

    /**
     * @brief Sets the stroke width for the path.
     *
     * This function defines the thickness of the stroke applied to all figures
     * in the path object. A stroke is the outline drawn along the edges of the
     * path's geometry.
     *
     * @param[in] width The width of the stroke in pixels. Must be positive value. (The default is 0)
     *
     * @note A value of @p width 0 disables the stroke.
     *
     * @see strokeFill()
     */
    Result strokeWidth(float width) noexcept;

    /**
     * @brief Sets the stroke color for the path.
     *
     * This function defines the RGBA color of the stroke applied to all figures
     * in the path object. The stroke color is used when rendering the outline
     * of the path geometry.
     *
     * @param[in] r The red color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] g The green color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] b The blue color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] a The alpha channel value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque. The default value is 0.
     *
     * @note If the stroke width is 0 (default), the stroke will not be visible regardless of the color.
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     *
     * @see strokeWidth()
     * @see strokeFill(Fill* f)
     */
    Result strokeFill(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) noexcept;

    /**
     * @brief Sets the gradient fill of the stroke for all of the figures from the path.
     *
     * @param[in] f The gradient fill.
     *
     * @retval Result::InvalidArgument In case a @c nullptr is passed as the argument.
     *
     * @note If the stroke width is 0 (default), the stroke will not be visible regardless of the color.
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     *
     * @see strokeFill(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
     */
    Result strokeFill(Fill* f) noexcept;

    /**
     * @brief Sets the dash pattern of the stroke.
     *
     * @param[in] dashPattern An array of alternating dash and gap lengths.
     * @param[in] cnt The length of the @p dashPattern array.
     * @param[in] offset The shift of the starting point within the repeating dash pattern, from which the pattern begins to be applied.
     *
     * @retval Result::InvalidArguments In case @p dashPattern is @c nullptr and @p cnt > 0 or @p dashPattern is not @c nullptr and @p cnt is zero.
     *
     * @note To reset the stroke dash pattern, pass @c nullptr to @p dashPattern and zero to @p cnt.
     * @note Values of @p dashPattern less than zero are treated as zero.
     * @note If all values in the @p dashPattern are equal to or less than 0, the dash is ignored.
     * @note If the @p dashPattern contains an odd number of elements, the sequence is repeated in the same
     * order to form an even-length pattern, preserving the alternation of dashes and gaps.
     *
     * @since 1.0
     */
    Result strokeDash(const float* dashPattern, uint32_t cnt, float offset = 0.0f) noexcept;

    /**
     * @brief Sets the cap style of the stroke in the open sub-paths.
     *
     * @param[in] cap The cap style value. The default value is @c StrokeCap::Square.
     *
     */
    Result strokeCap(StrokeCap cap) noexcept;

    /**
     * @brief Sets the join style for stroked path segments.
     *
     * The join style is used for joining the two line segment while stroking the path.
     *
     * @param[in] join The join style value. The default value is @c StrokeJoin::Bevel.
     *
     */
    Result strokeJoin(StrokeJoin join) noexcept;

    /**
     * @brief Sets the stroke miterlimit.
     *
     * @param[in] miterlimit The miterlimit imposes a limit on the extent of the stroke join, when the @c StrokeJoin::Miter join style is set. The default value is 4.
     *
     * @retval Result::InvalidArgument for @p miterlimit values less than zero.
     * 
     * @since 0.11
     */
    Result strokeMiterlimit(float miterlimit) noexcept;

    /**
     * @brief Sets the trim of the shape along the defined path segment, allowing control over which part of the shape is visible.
     *
     * If the values of the arguments @p begin and @p end exceed the 0-1 range, they are wrapped around in a manner similar to angle wrapping, effectively treating the range as circular.
     *
     * @param[in] begin Specifies the start of the segment to display along the path.
     * @param[in] end Specifies the end of the segment to display along the path.
     * @param[in] simultaneous Determines how to trim multiple paths within a single shape. If set to @c true (default), trimming is applied simultaneously to all paths;
     * Otherwise, all paths are treated as a single entity with a combined length equal to the sum of their individual lengths and are trimmed as such.
     *
     * @since 1.0
     */
    Result trimpath(float begin, float end, bool simultaneous = true) noexcept;

    /**
     * @brief Sets the solid color for all of the figures from the path.
     *
     * The parts of the shape defined as inner are colored.
     *
     * @param[in] r The red color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] g The green color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] b The blue color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] a The alpha channel value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque. The default value is 0.
     *
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     */
    Result fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) noexcept;

    /**
     * @brief Sets the gradient fill for all of the figures from the path.
     *
     * The parts of the shape defined as inner are filled.
     *
     * @param[in] f The unique pointer to the gradient fill.
     *
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     */
    Result fill(Fill* f) noexcept;

    /**
     * @brief Sets the fill rule for the shape.
     *
     * Specifies how the interior of the shape is determined when its path intersects itself.
     * The default fill rule is @c FillRule::NonZero.
     *
     * @param[in] r The fill rule to apply to the shape.
     */
    Result fillRule(FillRule r) noexcept;

    /**
     * @brief Sets the rendering order of the stroke and the fill.
     *
     * @param[in] strokeFirst If @c true the stroke is rendered before the fill, otherwise the stroke is rendered as the second one (the default option).
     *
     * @since 0.10
     */
    Result order(bool strokeFirst) noexcept;

    /**
     * @brief Retrieves the current path data of the shape.
     *
     * This function provides access to the shape's path data, including the commands
     * and points that define the path.
     *
     * @param[out] cmds Pointer to the array of commands representing the path.
     *                  Can be @c nullptr if this information is not needed.
     * @param[out] cmdsCnt Pointer to the variable that receives the number of commands in the @p cmds array.
     *                     Can be @c nullptr if this information is not needed.
     * @param[out] pts Pointer to the array of two-dimensional points that define the path.
     *                 Can be @c nullptr if this information is not needed.
     * @param[out] ptsCnt Pointer to the variable that receives the number of points in the @p pts array.
     *                    Can be @c nullptr if this information is not needed.
     */
    Result path(const PathCommand** cmds, uint32_t* cmdsCnt, const Point** pts, uint32_t* ptsCnt) const noexcept;

    /**
     * @brief Gets the pointer to the gradient fill of the shape.
     *
     * @return The pointer to the gradient fill of the stroke when succeed, @c nullptr in case no fill was set.
     */
    const Fill* fill() const noexcept;

    /**
     * @brief Gets the solid color of the shape.
     *
     * @param[out] r The red color channel value in the range [0 ~ 255].
     * @param[out] g The green color channel value in the range [0 ~ 255].
     * @param[out] b The blue color channel value in the range [0 ~ 255].
     * @param[out] a The alpha channel value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque.
     *
     */
    Result fill(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a = nullptr) const noexcept;

    /**
     * @brief Retrieves the current fill rule used by the shape.
     *
     * This function returns the fill rule, which determines how the interior 
     * regions of the shape are calculated when it overlaps itself.
     *
     * @see Shape::fillRule(FillRule r)
     * @return The current FillRule value of the shape.
     */
    FillRule fillRule() const noexcept;

    /**
     * @brief Gets the stroke width.
     *
     * @return The stroke width value when succeed, zero if no stroke was set.
     */
    float strokeWidth() const noexcept;

    /**
     * @brief Gets the color of the shape's stroke.
     *
     * @param[out] r The red color channel value in the range [0 ~ 255].
     * @param[out] g The green color channel value in the range [0 ~ 255].
     * @param[out] b The blue color channel value in the range [0 ~ 255].
     * @param[out] a The alpha channel value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque.
     *
     */
    Result strokeFill(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a = nullptr) const noexcept;

    /**
     * @brief Gets the pointer to the gradient fill of the stroke.
     *
     * @return The pointer to the gradient fill of the stroke when succeed, @c nullptr otherwise.
     */
    const Fill* strokeFill() const noexcept;

    /**
     * @brief Gets the dash pattern of the stroke.
     *
     * @param[out] dashPattern The pointer to the memory, where the dash pattern array is stored.
     * @param[out] offset The shift of the starting point within the repeating dash pattern.
     *
     * @return The length of the @p dashPattern array.
     *
     * @since 1.0
     */
    uint32_t strokeDash(const float** dashPattern, float* offset = nullptr) const noexcept;

    /**
     * @brief Gets the cap style used for stroking the path.
     *
     * @return The cap style value of the stroke.
     */
    StrokeCap strokeCap() const noexcept;

    /**
     * @brief Gets the join style value used for stroking the path.
     *
     * @return The join style value of the stroke.
     */
    StrokeJoin strokeJoin() const noexcept;

    /**
     * @brief Gets the stroke miterlimit.
     *
     * @return The stroke miterlimit value when succeed, 4 if no stroke was set.
     *
     * @since 0.11
     */
    float strokeMiterlimit() const noexcept;

    /**
     * @brief Creates a new Shape object.
     *
     * This function allocates and returns a new Shape instance.
     * To properly destroy the Shape object, use @ref Paint::rel().
     *
     * @return A pointer to the newly created Shape object.
     *
     * @see Paint::rel()
     */
    static Shape* gen() noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the Shape instance.
     *
     * @since 1.0
     */
    Type type() const noexcept override;

    _TVG_DECLARE_PRIVATE_DERIVE(Shape);
};


/**
 * @class Picture
 *
 * @brief A class representing an image read in one of the supported formats: raw, svg, png, jpg, lot and etc.
 * Besides the methods inherited from the Paint, it provides methods to load & draw images on the canvas.
 *
 * @note Supported formats are depended on the available TVG loaders.
 * @note See Animation class if the picture data is animatable.
 *
 * @warning This class is not designed for inheritance.
 */
struct TVG_API Picture : Paint
{
    /**
     * @brief Loads a picture data directly from a file.
     *
     * ThorVG efficiently caches the loaded data using the specified @p path as a key.
     * This means that loading the same file again will not result in duplicate operations;
     * instead, ThorVG will reuse the previously loaded picture data.
     *
     * @param[in] filename A file name, including the path, for the picture file.
     *
     * @retval Result::InvalidArguments In case the @p path is invalid.
     * @retval Result::NonSupport When trying to load a file with an unknown extension.
     *
     * @note The Load behavior can be asynchronous if the assigned thread number is greater than zero.
     * @see Initializer::init()
     */
    Result load(const char* filename) noexcept;

    /**
     * @brief Loads a picture data from a memory block of a given size.
     *
     * ThorVG efficiently caches the loaded data using the specified @p data address as a key
     * when the @p copy has @c false. This means that loading the same data again will not result in duplicate operations
     * for the sharable @p data. Instead, ThorVG will reuse the previously loaded picture data.
     *
     * @param[in] data A pointer to a memory location where the content of the picture file is stored. A null-terminated string is expected for non-binary data if @p copy is @c false.
     * @param[in] size The size in bytes of the memory occupied by the @p data.
     * @param[in] mimeType Mimetype or extension of data such as "jpg", "jpeg", "lot", "lottie+json", "svg", "svg+xml", "png", etc. In case an empty string or an unknown type is provided, the loaders will be tried one by one.
     * @param[in] rpath A resource directory path, if the @p data needs to access any external resources.
     * @param[in] copy If @c true the data are copied into the engine local buffer, otherwise they are not.
     *
     * @retval Result::InvalidArguments In case no data are provided or the @p size is zero or less.
     * @retval Result::NonSupport When trying to load a file with an unknown extension.
     *
     * @warning It's the user responsibility to release the @p data memory.
     *
     * @note If you are unsure about the MIME type, you can provide an empty value like @c nullptr, and thorvg will attempt to figure it out.
     * @since 0.5
     */
    Result load(const char* data, uint32_t size, const char* mimeType, const char* rpath = nullptr, bool copy = false) noexcept;

    /**
     * @brief Resizes the picture content to the given width and height.
     *
     * The picture content is resized while keeping the default size aspect ratio.
     * The scaling factor is established for each of dimensions and the smaller value is applied to both of them.
     *
     * @param[in] w A new width of the image in pixels.
     * @param[in] h A new height of the image in pixels.
     *
     */
    Result size(float w, float h) noexcept;

    /**
     * @brief Gets the size of the image.
     *
     * @param[out] w The width of the image in pixels.
     * @param[out] h The height of the image in pixels.
     *
     */
    Result size(float* w, float* h) const noexcept;

    /**
     * @brief Sets the normalized origin point of the Picture object.
     *
     * This method defines the origin point of the Picture using normalized coordinates.
     * Unlike a typical pivot point used only for transformations, this origin affects both
     * the transformation behavior and the actual rendering position of the Picture.
     *
     * The specified origin becomes the reference point for positioning the Picture on the canvas.
     * For example, setting the origin to (0.5f, 0.5f) moves the visual center of the picture
     * to the position specified by Paint::translate().
     *
     * The coordinates are given in a normalized range relative to the picture's bounds:
     * - (0.0f, 0.0f): top-left corner
     * - (0.5f, 0.5f): center
     * - (1.0f, 1.0f): bottom-right corner
     *
     * @param[in] x The normalized x-coordinate of the origin point (range: 0.0f to 1.0f).
     * @param[in] y The normalized y-coordinate of the origin point (range: 0.0f to 1.0f).
     *
     * @note This origin directly affects how the Picture is placed on the canvas when using
     *       transformations such as translate(), rotate(), or scale().
     *
     * @see Paint::translate()
     * @see Paint::rotate()
     * @see Paint::scale()
     *
     * @since 1.0
     */
    Result origin(float x, float y) noexcept;

    /**
     * @brief Gets the normalized origin point of the Picture object.
     *
     * This method retrieves the current origin point of the Picture, expressed
     * in normalized coordinates relative to the picture’s bounds.
     *
     * @param[out] x The normalized x-coordinate of the origin (range: 0.0f to 1.0f).
     * @param[out] y The normalized y-coordinate of the origin (range: 0.0f to 1.0f).
     *
     * @see origin()
     * @since 1.0
     */
    Result origin(float* x, float* y) const noexcept;

    /**
     * @brief Loads raw image data in a specific format from a memory block of the given size.
     *
     * ThorVG efficiently caches the loaded data, using the provided @p data address as a key
     * when @p copy is set to @c false. This allows ThorVG to avoid redundant operations
     * by reusing the previously loaded picture data for the same sharable @p data,
     * rather than duplicating the load process.
     *
     * @param[in] data A pointer to the memory block where the raw image data is stored.
     * @param[in] w The width of the image in pixels.
     * @param[in] h The height of the image in pixels.
     * @param[in] cs Specifies how the 32-bit color values should be interpreted.
     * @param[in] copy If @c true, the data is copied into the engine's local buffer. If @c false, the data is not copied.
     *
     * @since 0.9
     */
    Result load(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, bool copy = false) noexcept;

    /**
     * @brief Sets the asset resolver callback for handling external resources (e.g., images and fonts).
     *
     * This callback is invoked when an external asset reference (such as an image source or file path)
     * is encountered in a Picture object. It allows the user to provide a custom mechanism for loading
     * or substituting assets, such as loading from an external source or a virtual filesystem.
     *
     * @param[in] func A user-defined function that handles the resolution of asset paths.
     *                 @p resolver can return @c true if the asset was successfully resolved by the user, or @c false if it was not.
     * @param[in] data A pointer to user-defined data that will be passed to the callback each time it is invoked.
     *                 This can be used to maintain context or access external resources.
     *
     * @retval Result::InsufficientCondition If the picture is already loaded.
     * 
     * @warning This function must be called before @ref Picture::load()
     *          Setting the resolver after loading will have no effect on asset resolution for that asset.
     * @note @p src will be either a font path or a font name. In the case of a font name, @p src will begin with "name:", e.g., "name:FreeSans-Medium".
     * @note If @c false is returned by @p func, ThorVG will attempt to resolve the resource using its internal resolution mechanism as a fallback.
     * @note To unset the resolver, pass @c nullptr as the @p func parameter.
     *
     * @note Experimental API
     */
    Result resolver(std::function<bool(Paint* paint, const char* src, void* data)> func, void* data) noexcept;

    /**
     * @brief Retrieve a paint object from the Picture scene by its Unique ID.
     *
     * This function searches for a paint object within the Picture scene that matches the provided @p id.
     *
     * @param[in] id The Unique ID of the paint object.
     *
     * @return A pointer to the paint object that matches the given identifier, or @c nullptr if no matching paint object is found.
     *
     * @see Accessor::id()
     *
     * @since 1.0
     */
    const Paint* paint(uint32_t id) noexcept;

    /**
     * @brief Creates a new Picture object.
     *
     * This function allocates and returns a new Picture instance.
     * To properly destroy the Picture object, use @ref Paint::rel().
     *
     * @return A pointer to the newly created Picture object.
     *
     * @see Paint::rel()
     */
    static Picture* gen() noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the Picture instance.
     *
     * @since 1.0
     */
    Type type() const noexcept override;

    _TVG_DECLARE_ACCESSOR(Animation);
    _TVG_DECLARE_PRIVATE_DERIVE(Picture);
};


/**
 * @class Scene
 *
 * @brief A class to composite children paints.
 *
 * As the traditional graphics rendering method, TVG also enables scene-graph mechanism.
 * This feature supports an array function for managing the multiple paints as one group paint.
 *
 * As a group, the scene can be transformed, made translucent and composited with other target paints,
 * its children will be affected by the scene world.
 *
 * @warning This class is not designed for inheritance.
 */
struct TVG_API Scene : Paint
{
    /**
     * @brief Adds a paint object to the scene.
     *
     * Appends a paint object to the scene. If the optional @p at parameter is provided, 
     * the paint object is inserted immediately before the specified paint in the scene. 
     * If @p at is @c nullptr, the paint object is appended to the end of the scene.
     *
     * @param[in] target A pointer to the Paint object to be added to the scene.
     *                   This parameter must not be @c nullptr.
     * @param[in] at     A pointer to an existing Paint object in the scene before
     *                   which @p target will be inserted. If @c nullptr,
     *                   @p target is appended to the end of the scene.
     *                   The default value is @c nullptr.
     *
     * @note Ownership of the @p target object is transferred to the scene upon
     *       successful addition. To retain ownership, call @ref Paint::ref()
     *       before adding it to the scene.
     * @note The rendering order of paint objects follows their order in the scene.
     *       If layering is required, ensure the paints are added in the desired order.
     *
     * @see Scene::paints()
     * @see Scene::remove()
     * @see Paint::ref()
     *
     * @since 1.0
     */
    Result add(Paint* target, Paint* at = nullptr) noexcept;

    /**
     * @brief Returns the list of paints currently held by the Scene.
     *
     * This function provides a list of paint nodes, allowing users to access scene-graph information.
     *
     * @see Scene::add()
     * @see Scene::remove()
     *
     * @warning This is read-only. Do not modify the list.
     * @since 1.0
     */
    const std::list<Paint*>& paints() const noexcept;

    /**
     * @brief Removes a paint object or all paint objects from the scene.
     *
     * This function removes a specified paint object from the scene. If no paint
     * object is specified (i.e., the default @c nullptr is used), the function
     * performs to clear all paints from the scene.
     *
     * @param[in] paint A pointer to the Paint object to be removed from the scene.
     *                  If @c nullptr, remove all the paints from the scene.
     *
     * @see Scene::add()
     * @see Scene::paints()
     *
     * @since 1.0
     */
    Result remove(Paint* paint = nullptr) noexcept;

    /**
     * @brief Add a post-processing effect to the scene.
     *
     * Adds a post-processing effect to the scene's effect pipeline, which is applied
     * after the scene has been rendered. Effects are applied cumulatively and in the
     * order they are added. Calling this function multiple times will chain multiple
     * effects sequentially.
     *
     * Certain effects may be used to modify the pipeline behavior itself. For example,
     * @c SceneEffect::Clear removes all previously added effects.
     *
     * @param[in] effect The scene effect to add. Available effects are defined in the @ref SceneEffect enum.
     * @param[in] ...    Additional variadic parameters required for certain effects (e.g., sigma and direction for GaussianBlur).
     *
     * @note The caller must provide the correct parameters for the selected effect.
     *       Supplying incorrect or insufficient arguments results in undefined behavior.
     *
     * @since 1.0
     */
    Result add(SceneEffect effect, ...) noexcept;

    /**
     * @brief Creates a new Scene object.
     *
     * This function allocates and returns a new Scene instance.
     * To properly destroy the Scene object, use @ref Paint::rel().
     *
     * @return A pointer to the newly created Scene object.
     *
     * @see Paint::rel()
     */
    static Scene* gen() noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the Scene instance.
     *
     * @since 1.0
     */
    Type type() const noexcept override;

    _TVG_DECLARE_PRIVATE_DERIVE(Scene);
};


/**
 * @class Text
 *
 * @brief A class to represent text objects in a graphical context, allowing for rendering and manipulation of unicode text.
 *
 * @warning This class is not designed for inheritance.
 *
 * @since 0.15
 */
struct TVG_API Text : Paint
{
    /**
     * @brief Sets the font family for the text.
     *
     * This function specifies the name of the font to be used when rendering text.
     *
     * @param[in] name The name of the font. This should match a font available through the canvas backend.
     *                 If set to @c nullptr, ThorVG will attempt to select a fallback font available on the engine.
     *
     * @retval Result::InsufficientCondition when the specified @p name cannot be found.
     *
     * @note This function only sets the font family name. Use @ref size() to define the font size.
     * @note If the @p name is not specified, ThorVG will select an available fallback font.
     *
     * @see Text::size()
     * @see Text::load()
     *
     * @since 1.0
     */
    Result font(const char* name) noexcept;

    /**
     * @brief Sets the font size for the text.
     *
     * This function sets the font size used during text rendering.
     * The size is specified in point units, and supports floating-point precision
     * for smooth scaling and animation effects.
     *
     * @param[in] size The font size in points. Must be greater than 0.0.
     *
     * @retval Result::InvalidArguments if the @p size is less than or equal to 0.
     *
     * @note Use this function in combination with @ref font() to fully define text appearance.
     * @note Fractional sizes (e.g., 12.5) are supported for sub-pixel rendering and animations.
     *
     * @see Text::font()
     *
     * @since 1.0
     */
    Result size(float size) noexcept;

    /**
     * @brief Assigns the given unicode text to be rendered.
     *
     * This function sets the unicode string that will be displayed by the rendering system.
     * The text is set according to the specified UTF encoding method, which defaults to UTF-8.
     *
     * @param[in] text The multi-byte text encoded with utf8 string to be rendered.
     *
     * @since 1.0
     */
    Result text(const char* text) noexcept;

    /**
     * @brief Sets text alignment or anchor per axis.
     *
     * If layout width/height is set on an axis, align within the layout box.
     * Otherwise, treat it as an anchor within the text bounds which point of
     * the text box is pinned to the paint position.
     *
     * @param[in] x Horizontal alignment/anchor in [0..1]: 0=left/start, 0.5=center, 1=right/end. (Default is 0)
     * @param[in] y Vertical alignment/anchor in [0..1]: 0=top, 0.5=middle, 1=bottom. (Default is 0)
     *
     * @since 1.0
     *
     * @see layout()
     */
    Result align(float x, float y) noexcept;

    /**
     * @brief Sets the virtual layout box (constraints) for the text.
     *
     * If width/height is set on an axis, that axis is constrained by a virtual layout box and
     * the text may wrap/align inside it. If width/height == 0, the axis is
     * unconstrained and @ref align() acts as an anchor on that axis.
     *
     * @param[in] w Layout width in user space. Use 0 for no horizontal constraint. (Default is 0)
     * @param[in] h Layout height in user space. Use 0 for no vertical constraint. (Default is 0)
     *
     * @note This defines constraints only; alignment/anchoring is controlled by @ref align().
     * @since 1.0
     *
     * @see align()
     * @see spacing()
     */
    Result layout(float w, float h) noexcept;

    /**
     * @brief Sets the text wrapping mode for this text object.
     *
     * This method controls how the text is laid out when it exceeds the available space.
     * The wrapping mode determines whether text is truncated, wrapped by character or word,
     * or adjusted automatically. An ellipsis mode is also available for truncation with "...".
     *
     * @param[in] mode The wrapping strategy to apply. Default is @c TextWrap::None
     *
     * @see TextWrap
     * @since 1.0
     */
    Result wrap(TextWrap mode) noexcept;

    /**
     * @brief Apply an italic (slant) transformation to the text.
     *
     * This function applies a shear transformation to simulate an italic (oblique) style
     * for the current text object. The shear factor determines the degree of slant
     * applied along the X-axis.
     *
     * @param[in] shear The shear factor to apply. A value of 0.0 applies no slant, while values around 0.5 result in a strong slant.
     *                  Must be in the range [0.0, 0.5]. Default value is 0.18.
     *
     * @note The @p shear factor will be clamped to the valid range if it exceeds the limits.
     * @note This does not require the font itself to be italic.
     *       It visually simulates the effect by applying a transformation matrix.
     *
     * @warning Excessive slanting may cause visual distortion depending on the font and size.
     *
     * @see Text::font()
     *
     * @since 1.0
     */
    Result italic(float shear = 0.18f) noexcept;

    /**
     * @brief Sets an outline (stroke) around the text object.
     *
     * This function adds an outline to the text with the specified width and RGB color.
     * The outline enhances the visibility of the text by rendering a stroke around its glyphs.
     *
     * @param width The width of the outline. Must be positive value. (The default is 0)
     * @param r     Red component of the outline color (0–255).
     * @param g     Green component of the outline color (0–255).
     * @param b     Blue component of the outline color (0–255).
     *
     * @note To disable the outline, set @p width to 0.
     * @see Text::fill() to set the main text fill color.
     *
     * @since 1.0
     */
    Result outline(float width, uint8_t r, uint8_t g, uint8_t b) noexcept;

    /**
     * @brief Sets the text color.
     *
     * @param[in] r The red color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] g The green color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] b The blue color channel value in the range [0 ~ 255]. The default value is 0.
     *
     * @see Text::font()
     * @see Text::outline()
     *
     * @since 0.15
     */
    Result fill(uint8_t r, uint8_t g, uint8_t b) noexcept;

    /**
     * @brief Sets the gradient fill for all of the figures from the text.
     *
     * The parts of the text defined as inner are filled.
     *
     * @param[in] f The unique pointer to the gradient fill.
     *
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     * @see Text::font()
     *
     * @since 0.15
     */
    Result fill(Fill* f) noexcept;

    /**
     * @brief Set the spacing scale factors for text layout.
     *
     * This function adjusts the letter spacing (horizontal space between glyphs) and
     * line spacing (vertical space between lines of text) using scale factors.
     *
     * Both values are relative to the font's default metrics:
     * - The letter spacing is applied as a scale factor to the glyph's advance width.
     * - The line spacing is applied as a scale factor to the glyph's advance height.
     *
     * @param[in] letter The scale factor for letter spacing.
     *                   Values > 1.0 increase spacing, values < 1.0 decrease it.
     *                   Must be greater than or equal to 0.0. (default: 1.0)
     *
     * @param[in] line The scale factor for line spacing.
     *                 Values > 1.0 increase line spacing, values < 1.0 decrease it.
     *                 Must be greater than or equal to 0.0. (default: 1.0)
     *
     * @since 1.0
     */
    Result spacing(float letter, float line) noexcept;

    /**
     * @brief Retrieves the layout metrics of the text object.
     *
     * Fills the provided `TextMetrics` structure with the font layout values of this text object,
     * such as ascent, descent, linegap, and line advance.
     *
     * The returned values reflect the font size applied to the text object,
     * but do not include any transformations (e.g., scale, rotation, or translation).
     *
     * @param[out] metrics A reference to a `TextMetrics` structure to be filled with the resulting values.
     *
     * @return Result::InsufficientCondition if no font or size has been set yet.
     *
     * @see TextMetrics
     * @note Experimental API
     */
    Result metrics(TextMetrics& metrics) const noexcept;

    /**
     * @brief Loads a scalable font data (ttf) from a file.
     *
     * ThorVG efficiently caches the loaded data using the specified @p path as a key.
     * This means that loading the same file again will not result in duplicate operations;
     * instead, ThorVG will reuse the previously loaded font data.
     *
     * @param[in] filename A file name, including the path, for the font file.
     *
     * @retval Result::InvalidArguments In case the @p path is invalid.
     * @retval Result::NonSupport When trying to load a file with an unknown extension.
     *
     * @see Text::unload(const char* filename)
     *
     * @since 0.15
     */
    static Result load(const char* filename) noexcept;

    /**
     * @brief Loads a scalable font data (ttf) from a memory block of a given size.
     *
     * ThorVG efficiently caches the loaded font data using the specified @p name as a key.
     * This means that loading the same fonts again will not result in duplicate operations.
     * Instead, ThorVG will reuse the previously loaded font data.
     *
     * @param[in] name The name under which the font will be stored and accessible (e.x. in a @p font() API).
     * @param[in] data A pointer to a memory location where the content of the font data is stored.
     * @param[in] size The size in bytes of the memory occupied by the @p data.
     * @param[in] mimeType Mimetype or extension of font data. In case an empty string is provided the loader will be determined automatically.
     * @param[in] copy If @c true the data are copied into the engine local buffer, otherwise they are not (default).
     *
     * @retval Result::InvalidArguments If no name is provided or if @p size is zero while @p data points to a valid memory location.
     * @retval Result::NonSupport When trying to load a file with an unsupported extension.
     * @retval Result::InsufficientCondition If attempting to unload the font data that has not been previously loaded.
     *
     * @warning It's the user responsibility to release the @p data memory.
     *
     * @note To unload the font data loaded using this API, pass the proper @p name and @c nullptr as @p data.
     * @note If you are unsure about the MIME type, you can provide an empty value like @c nullptr, and thorvg will attempt to figure it out.
     * @see Text::font(const char* name, float size, const char* style)
     *
     * @since 0.15
     */
    static Result load(const char* name, const char* data, uint32_t size, const char* mimeType = "ttf", bool copy = false) noexcept;

    /**
     * @brief Unloads the specified scalable font data (TTF) that was previously loaded.
     *
     * This function is used to release resources associated with a font file that has been loaded into memory.
     *
     * @param[in] filename The file name of the loaded font, including the path.
     *
     * @retval Result::InsufficientCondition Fails if the loader is not initialized.
     *
     * @note If the font data is currently in use, it will not be immediately unloaded.
     * @see Text::load(const char* filename)
     * 
     * @since 0.15
     */
    static Result unload(const char* filename) noexcept;

    /**
     * @brief Creates a new Text object.
     *
     * This function allocates and returns a new Text instance.
     * To properly destroy the Text object, use @ref Paint::rel().
     *
     * @return A pointer to the newly created Text object.
     *
     * @see Paint::rel()
     *
     * @since 0.15
     */
    static Text* gen() noexcept;

    /**
     * @brief Returns the ID value of this class.
     *
     * This method can be used to check the current concrete instance type.
     *
     * @return The class type ID of the Text instance.
     *
     * @since 1.0
     */
    Type type() const noexcept override;

    _TVG_DECLARE_PRIVATE_DERIVE(Text);
};


/**
 * @class SwCanvas
 *
 * @brief A class for the rendering graphical elements with a software raster engine.
 */
struct TVG_API SwCanvas final : Canvas
{
    ~SwCanvas() override;

    /**
     * @brief Sets the drawing target for the rasterization.
     *
     * The buffer of a desirable size should be allocated and owned by the caller.
     *
     * @param[in] buffer A pointer to a memory block of the size @p stride x @p h, where the raster data are stored.
     * @param[in] stride The stride of the raster image - greater than or equal to @p w.
     * @param[in] w The width of the raster image.
     * @param[in] h The height of the raster image.
     * @param[in] cs The value specifying the way the 32-bits colors should be read/written.
     *
     * @retval Result::InvalidArguments In case no valid pointer is provided or the width, or the height or the stride is zero.
     * @retval Result::InsufficientCondition if the canvas is performing rendering. Please ensure the canvas is synced.
     * @retval Result::NonSupport In case the software engine is not supported.
     *
     * @warning Do not access @p buffer during Canvas::add() - Canvas::sync(). It should not be accessed while the engine is writing on it.
     *
     * @see Canvas::viewport()
     * @see Canvas::sync()
    */
    Result target(uint32_t* buffer, uint32_t stride, uint32_t w, uint32_t h, ColorSpace cs) noexcept;

    /**
     * @brief Creates a new SwCanvas object with optional rendering engine settings.
     *
     * This method generates a software canvas instance that can be used for drawing vector graphics.
     * It accepts an optional parameter @p op to choose between different rendering engine behaviors.
     *
     * @param[in] op The rendering engine option. Default is @c EngineOption::Default.
     *
     * @return A new SwCanvas object.
     *
     * @see enum EngineOption
     */
    static SwCanvas* gen(EngineOption op = EngineOption::Default) noexcept;

    _TVG_DECLARE_PRIVATE(SwCanvas);
};


/**
 * @class GlCanvas
 *
 * @brief A class for the rendering graphic elements with a GL raster engine.
 *
 * @since 0.14
 */
struct TVG_API GlCanvas final : Canvas
{
    ~GlCanvas() override;

    /**
     * @brief Sets the drawing target for rasterization.
     *
     * This function specifies the drawing target where the rasterization will occur. It can target
     * a specific framebuffer object (FBO) or the main surface.
     *
     * @param[in] display The platform-specific display handle (EGLDisplay for EGL). Set @c nullptr for other systems.
     * @param[in] surface The platform-specific surface handle (EGLSurface for EGL, HDC for WGL). Set @c nullptr for other systems.
     * @param[in] context The OpenGL context to be used for rendering on this canvas.
     * @param[in] id The GL target ID, usually indicating the FBO ID. A value of @c 0 specifies the main surface.
     * @param[in] w The width (in pixels) of the raster image.
     * @param[in] h The height (in pixels) of the raster image.
     * @param[in] cs Specifies how the pixel values should be interpreted. Currently, it only allows @c ColorSpace::ABGR8888S as @c GL_RGBA8.
     *
     * @retval Result::InsufficientCondition If the canvas is currently rendering.
     *         Ensure that @ref Canvas::sync() has been called before setting a new target.
     * @retval Result::NonSupport In case the gl engine is not supported.
     *
     * @note If @p display and @p surface are not provided, the ThorVG GL engine assumes that
     *       the appropriate OpenGL context is already current and will not attempt to bind a new one.
     *
     * @see Canvas::sync()
     *
     * @since 1.0
    */
    Result target(void* display, void* surface, void* context, int32_t id, uint32_t w, uint32_t h, ColorSpace cs) noexcept;

    /**
     * @brief Creates a new OpenGL/ES Canvas object with optional rendering engine settings.
     *
     * This method generates a OpenGL canvas instance that can be used for drawing vector graphics.
     * It accepts an optional parameter @p op to choose between different rendering engine behaviors.
     *
     * @param[in] op The rendering engine option. Default is @c EngineOption::Default.
     *
     * @return A new GlCanvas object.
     *
     * @note Currently, it does not support @c EngineOption::SmartRender. The request will be ignored.
     *
     * @see enum EngineOption
     *
     * @since 1.0
     */
    static GlCanvas* gen(EngineOption op = EngineOption::Default) noexcept;

    _TVG_DECLARE_PRIVATE(GlCanvas);
};


/**
 * @class WgCanvas
 *
 * @brief A class for the rendering graphic elements with a WebGPU raster engine.
 *
 * @warning Please do not use it. This class is not fully supported yet.
 *
 * @since 0.15
 */
struct TVG_API WgCanvas final : Canvas
{
    ~WgCanvas() override;

    /**
     * @brief Sets the drawing target for the rasterization.
     *
     * @param[in] device WGPUDevice, a desired handle for the wgpu device. If it is @c nullptr, ThorVG will assign an appropriate device internally.
     * @param[in] instance WGPUInstance, context for all other wgpu objects.
     * @param[in] target Either WGPUSurface or WGPUTexture, serving as handles to a presentable surface or texture.
     * @param[in] w The width of the target.
     * @param[in] h The height of the target.
     * @param[in] cs Specifies how the pixel values should be interpreted. Currently, it only allows @c ColorSpace::ABGR8888S as @c WGPUTextureFormat_RGBA8Unorm.
     * @param[in] type @c 0: surface, @c 1: texture are used as pesentable target.
     *
     * @retval Result::InsufficientCondition if the canvas is performing rendering. Please ensure the canvas is synced.
     * @retval Result::NonSupport In case the wg engine is not supported.
     *
     * @since 1.0
     *
     * @see Canvas::viewport()
     * @see Canvas::sync()
     */
    Result target(void* device, void* instance, void* target, uint32_t w, uint32_t h, ColorSpace cs, int type = 0) noexcept;

    /**
     * @brief Creates a new WebGPU Canvas object with optional rendering engine settings.
     *
     * This method generates a WebGPU canvas instance that can be used for drawing vector graphics.
     * It accepts an optional parameter @p op to choose between different rendering engine behaviors.
     *
     * @param[in] op The rendering engine option. Default is @c EngineOption::Default.
     *
     * @return A new WgCanvas object.
     *
     * @note Currently, it does not support @c EngineOption::SmartRender. The request will be ignored.
     *
     * @see enum EngineOption
     *
     * @since 1.0
     */
    static WgCanvas* gen(EngineOption op = EngineOption::Default) noexcept;

    _TVG_DECLARE_PRIVATE(WgCanvas);
};


/**
 * @class Initializer
 *
 * @brief A class that enables initialization and termination of the TVG engines.
 */
struct TVG_API Initializer final
{
    /**
     * @brief Initializes the ThorVG engine runtime.
     *
     * ThorVG requires an active runtime environment for rendering operations.
     * This function sets up an internal task scheduler and creates a specified number
     * of worker threads to enable parallel rendering.
     *
     * @param[in] threads The number of worker threads to launch.
     *                    A value of 0 indicates that only the main thread will be used.
     *
     * @return Result indicating success or failure of initialization.
     *
     * @note This function uses internal reference counting to allow multiple init() calls.
     *       However, the number of threads is fixed during the first successful initialization
     *       and cannot be changed in subsequent calls.
     *
     * @see Initializer::term()
     */
    static Result init(uint32_t threads = 0) noexcept;

    /**
     * @brief Terminates the ThorVG engine.
     *
     * Cleans up resources and stops any internal threads initialized by init().
     *
     * @retval Result::InsufficientCondition Returned if there is nothing to terminate (e.g., init() was not called).
     *
     * @note The initializer maintains a reference count for safe repeated use.
     *       Only the final call to term() will fully shut down the engine.
     * @see Initializer::init()
     */
    static Result term() noexcept;

    /**
     * @brief Retrieves the version of the TVG engine.
     *
     * @param[out] major A major version number.
     * @param[out] minor A minor version number.
     * @param[out] micro A micro version number.
     *
     * @return The version of the engine in the format major.minor.micro, or a @p nullptr in case of an internal error.
     *
     * @since 0.15
     */
    static const char* version(uint32_t* major, uint32_t* minor, uint32_t* micro) noexcept;

    _TVG_DISABLE_CTOR(Initializer);
};


/**
 * @class Animation
 *
 * @brief The Animation class enables manipulation of animatable images.
 *
 * This class supports the display and control of animation frames.
 *
 * @since 0.13
 */
struct TVG_API Animation
{
    virtual ~Animation();

    /**
     * @brief Specifies the current frame in the animation.
     *
     * @param[in] no The index of the animation frame to be displayed. The index should be less than the totalFrame().
     *
     * @retval Result::InsufficientCondition if the given @p no is the same as the current frame value.
     * @retval Result::NonSupport The current Picture data does not support animations.
     *
     * @note For efficiency, ThorVG ignores updates to the new frame value if the difference from the current frame value
     *       is less than 0.001. In such cases, it returns @c Result::InsufficientCondition.
     *       Values less than 0.001 may be disregarded and may not be accurately retained by the Animation.
     *
     * @see totalFrame()
     */
    Result frame(float no) noexcept;

    /**
     * @brief Retrieves a picture instance associated with this animation instance.
     *
     * This function provides access to the picture instance that can be used to load animation formats, such as lot.
     * After setting up the picture, it can be added to the designated canvas, enabling control over animation frames
     * with this Animation instance.
     *
     * @return A picture instance that is tied to this animation.
     *
     * @warning The picture instance is owned by Animation. It should not be deleted manually.
     */
    Picture* picture() const noexcept;

    /**
     * @brief Retrieves the current frame number of the animation.
     *
     * @return The current frame number of the animation, between 0 and totalFrame() - 1.
     *
     * @note If the Picture is not properly configured, this function will return 0.
     *
     * @see Animation::frame()
     * @see Animation::totalFrame()
     */
    float curFrame() const noexcept;

    /**
     * @brief Retrieves the total number of frames in the animation.
     *
     * @return The total number of frames in the animation.
     *
     * @note Frame numbering starts from 0.
     * @note If the Picture is not properly configured, this function will return 0.
     */
    float totalFrame() const noexcept;

    /**
     * @brief Retrieves the duration of the animation in seconds.
     *
     * @return The duration of the animation in seconds.
     *
     * @note If the Picture is not properly configured, this function will return 0.
     */
    float duration() const noexcept;

    /**
     * @brief Specifies the playback segment of the animation.
     *
     * The set segment is designated as the play area of the animation.
     * This is useful for playing a specific segment within the entire animation.
     * After setting, the number of animation frames and the playback time are calculated
     * by mapping the playback segment as the entire range.
     *
     * @param[in] begin segment begin frame.
     * @param[in] end segment end frame.
     *
     * @retval Result::InsufficientCondition In case the animation is not loaded.
     * @retval Result::InvalidArguments If the @p begin is higher than @p end.
     * @retval Result::NonSupport When it's not animatable.
     *
     * @note Animation allows a range from 0.0 to the total frame. @p end should not be higher than @p begin.
     * @note If a marker has been specified, its range will be disregarded.
     *
     * @see LottieAnimation::segment(const char* marker)
     * @see Animation::totalFrame()
     *
     * @since 1.0
     */
    Result segment(float begin, float end) noexcept;

    /**
     * @brief Gets the current segment range information.
     *
     * @param[out] begin segment begin frame.
     * @param[out] end segment end frame.
     *
     * @retval Result::InsufficientCondition In case the animation is not loaded.
     * @retval Result::NonSupport When it's not animatable.
     *
     * @since 1.0
     */
    Result segment(float* begin, float* end = nullptr) noexcept;

    /**
     * @brief Creates a new Animation object.
     *
     * @return A new Animation object.
     */
    static Animation* gen() noexcept;

    _TVG_DECLARE_PRIVATE_BASE(Animation);
};


/**
 * @class Saver
 *
 * @brief A class for exporting a paint object into a specified file, from which to recover the paint data later.
 *
 * ThorVG provides a feature for exporting & importing paint data. The Saver role is to export the paint data to a file.
 * It's useful when you need to save the composed scene or image from a paint object and recreate it later.
 *
 * The file format is decided by the extension name(i.e. "*.tvg") while the supported formats depend on the TVG packaging environment.
 * If it doesn't support the file format, the save() method returns the @c Result::NonSupport result.
 *
 * Once you export a paint to the file successfully, you can recreate it using the Picture class.
 *
 * @see Picture::load()
 *
 * @since 0.5
 */
struct TVG_API Saver final
{
    ~Saver();

    /**
     * @brief Sets the base background content for the saved image.
     *
     * @param[in] paint The paint to be drawn as the background image for the saving paint.
     *
     * @since 1.0
     */
    Result background(Paint* paint) noexcept;

    /**
     * @brief Exports the given @p paint data to the given @p path
     *
     * If the saver module supports any compression mechanism, it will optimize the data size.
     * This might affect the encoding/decoding time in some cases. You can turn off the compression
     * if you wish to optimize for speed.
     *
     * @param[in] paint The paint to be saved with all its associated properties.
     * @param[in] filename A file name, including the path, where the paint data will be saved.
     * @param[in] quality The encoded quality level. @c 0 is the minimum, @c 100 is the maximum value(recommended).
     *
     * @retval Result::InsufficientCondition If currently saving other resources.
     * @retval Result::NonSupport When trying to save a file with an unknown extension or in an unsupported format.
     * @retval Result::Unknown In case an empty paint is to be saved.
     *
     * @note Saving can be asynchronous if the assigned thread number is greater than zero. To guarantee the saving is done, call sync() afterwards.
     * @see Saver::sync()
     *
     * @since 0.5
     */
    Result save(Paint* paint, const char* filename, uint32_t quality = 100) noexcept;

    /**
     * @brief Export the provided animation data to the specified file path.
     *
     * This function exports the given animation data to the provided file path. You can also specify the desired frame rate in frames per second (FPS) by providing the fps parameter.
     *
     * @param[in] animation The animation to be saved, including all associated properties.
     * @param[in] filename A file name, including the path, where the animation will be saved.
     * @param[in] quality The encoded quality level. @c 0 is the minimum, @c 100 is the maximum value(recommended).
     * @param[in] fps The desired frames per second (FPS). For example, to encode data at 60 FPS, pass 60. Pass 0 to keep the original frame data.
     *
     * @retval Result::InsufficientCondition if there are ongoing resource-saving operations.
     * @retval Result::NonSupport if an attempt is made to save the file with an unknown extension or in an unsupported format.
     * @retval Result::Unknown if attempting to save an empty paint.
     *
     * @note A higher frames per second (FPS) would result in a larger file size. It is recommended to use the default value.
     * @note Saving can be asynchronous if the assigned thread number is greater than zero. To guarantee the saving is done, call sync() afterwards.
     *
     * @see Saver::sync()
     *
     * @since 1.0
     */
    Result save(Animation* animation, const char* filename, uint32_t quality = 100, uint32_t fps = 0) noexcept;

    /**
     * @brief Guarantees that the saving task is finished.
     *
     * The behavior of the Saver works on a sync/async basis, depending on the threading setting of the Initializer.
     * Thus, if you wish to have a benefit of it, you must call sync() after the save() in the proper delayed time.
     * Otherwise, you can call sync() immediately.
     *
     * @note The asynchronous tasking is dependent on the Saver module implementation.
     * @see Saver::save()
     *
     * @since 0.5
     */
    Result sync() noexcept;

    /**
     * @brief Creates a new Saver object.
     *
     * @return A new Saver object.
     *
     * @since 0.5
     */
    static Saver* gen() noexcept;

    _TVG_DECLARE_PRIVATE_BASE(Saver);
};


/**
 * @class Accessor
 *
 * @brief The Accessor is a utility class to debug the Scene structure by traversing the scene-tree.
 *
 * The Accessor helps you search specific nodes to read the property information, figure out the structure of the scene tree and its size.
 *
 * @warning We strongly warn you not to change the paints of a scene unless you really know the design-structure.
 *
 * @since 0.10
 */
struct TVG_API Accessor final
{
    ~Accessor();

    /**
     * @brief Set the access function for traversing the Picture scene tree nodes.
     *
     * @param[in] paint The paint node to traverse the internal scene-tree.
     * @param[in] func The callback function calling for every paint nodes of the Picture.
     * @param[in] data Data passed to the @p func as its argument.
     *
     * @note The bitmap based picture might not have the scene-tree.
     *
     * @since 1.0
     */
    Result set(Paint* paint, std::function<bool(const Paint* paint, void* data)> func, void* data) noexcept;

    /**
     * @brief Generate a unique ID (hash key) from a given name.
     *
     * This function computes a unique identifier value based on the provided string.
     * You can use this to assign a unique ID to the Paint object.
     *
     * @param[in] name The input string to generate the unique identifier from.
     *
     * @return The generated unique identifier value.
     *
     * @see Paint::id
     *
     * @since 1.0
     */
    static uint32_t id(const char* name) noexcept;

    /**
     * @brief Creates a new Accessor object.
     *
     * @return A new Accessor object.
     */
    static Accessor* gen() noexcept;

    _TVG_DECLARE_PRIVATE_BASE(Accessor);
};

/** @}*/

} //namespace

#endif //_THORVG_H_
