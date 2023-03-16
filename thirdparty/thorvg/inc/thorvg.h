/*!
 * @file thorvg.h
 *
 * The main APIs enabling the TVG initialization, preparation of the canvas and provisioning of its content:
 * - drawing shapes such as line, curve, arc, rectangle, circle or user-defined
 * - drawing pictures - SVG, PNG, JPG, RAW
 * - solid or gradient filling
 * - continuous and dashed stroking
 * - clipping and masking
 * and finally drawing the canvas and TVG termination.
 */


#ifndef _THORVG_H_
#define _THORVG_H_

#include <memory>
#include <string>

#ifdef TVG_BUILD
    #if defined(_WIN32) && !defined(__clang__)
        #define TVG_EXPORT __declspec(dllexport)
        #define TVG_DEPRECATED __declspec(deprecated)
    #else
        #define TVG_EXPORT __attribute__ ((visibility ("default")))
        #define TVG_DEPRECATED __attribute__ ((__deprecated__))
    #endif
#else
    #define TVG_EXPORT
    #define TVG_DEPRECATED
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define _TVG_DECLARE_PRIVATE(A) \
protected: \
    struct Impl; \
    Impl* pImpl; \
    A(const A&) = delete; \
    const A& operator=(const A&) = delete; \
    A()

#define _TVG_DISABLE_CTOR(A) \
    A() = delete; \
    ~A() = delete

#define _TVG_DECLARE_ACCESSOR() \
    friend Canvas; \
    friend Scene; \
    friend Picture; \
    friend Accessor; \
    friend IteratorAccessor


namespace tvg
{

class RenderMethod;
class IteratorAccessor;
class Scene;
class Picture;
class Canvas;
class Accessor;

/**
 * @defgroup ThorVG ThorVG
 * @brief ThorVG classes and enumerations providing C++ APIs.
 */

/**@{*/

/**
 * @brief Enumeration specifying the result from the APIs.
 */
enum class Result
{
    Success = 0,           ///< The value returned in case of a correct request execution.
    InvalidArguments,      ///< The value returned in the event of a problem with the arguments given to the API - e.g. empty paths or null pointers.
    InsufficientCondition, ///< The value returned in case the request cannot be processed - e.g. asking for properties of an object, which does not exist.
    FailedAllocation,      ///< The value returned in case of unsuccessful memory allocation.
    MemoryCorruption,      ///< The value returned in the event of bad memory handling - e.g. failing in pointer releasing or casting
    NonSupport,            ///< The value returned in case of choosing unsupported options.
    Unknown                ///< The value returned in all other cases.
};

/**
 * @brief Enumeration specifying the values of the path commands accepted by TVG.
 *
 * Not to be confused with the path commands from the svg path element (like M, L, Q, H and many others).
 * TVG interprets all of them and translates to the ones from the PathCommand values.
 */
enum class PathCommand
{
    Close = 0, ///< Ends the current sub-path and connects it with its initial point. This command doesn't expect any points.
    MoveTo,    ///< Sets a new initial point of the sub-path and a new current point. This command expects 1 point: the starting position.
    LineTo,    ///< Draws a line from the current point to the given point and sets a new value of the current point. This command expects 1 point: the end-position of the line.
    CubicTo    ///< Draws a cubic Bezier curve from the current point to the given point using two given control points and sets a new value of the current point. This command expects 3 points: the 1st control-point, the 2nd control-point, the end-point of the curve.
};

/**
 * @brief Enumeration determining the ending type of a stroke in the open sub-paths.
 */
enum class StrokeCap
{
    Square = 0, ///< The stroke is extended in both end-points of a sub-path by a rectangle, with the width equal to the stroke width and the length equal to the half of the stroke width. For zero length sub-paths the square is rendered with the size of the stroke width.
    Round,      ///< The stroke is extended in both end-points of a sub-path by a half circle, with a radius equal to the half of a stroke width. For zero length sub-paths a full circle is rendered.
    Butt        ///< The stroke ends exactly at each of the two end-points of a sub-path. For zero length sub-paths no stroke is rendered.
};

/**
 * @brief Enumeration determining the style used at the corners of joined stroked path segments.
 */
enum class StrokeJoin
{
    Bevel = 0, ///< The outer corner of the joined path segments is bevelled at the join point. The triangular region of the corner is enclosed by a straight line between the outer corners of each stroke.
    Round,     ///< The outer corner of the joined path segments is rounded. The circular region is centered at the join point.
    Miter      ///< The outer corner of the joined path segments is spiked. The spike is created by extension beyond the join point of the outer edges of the stroke until they intersect. In case the extension goes beyond the limit, the join style is converted to the Bevel style.
};

/**
 * @brief Enumeration specifying how to fill the area outside the gradient bounds.
 */
enum class FillSpread
{
    Pad = 0, ///< The remaining area is filled with the closest stop color.
    Reflect, ///< The gradient pattern is reflected outside the gradient area until the expected region is filled.
    Repeat   ///< The gradient pattern is repeated continuously beyond the gradient area until the expected region is filled.
};

/**
 * @brief Enumeration specifying the algorithm used to establish which parts of the shape are treated as the inside of the shape.
 */
enum class FillRule
{
    Winding = 0, ///< A line from the point to a location outside the shape is drawn. The intersections of the line with the path segment of the shape are counted. Starting from zero, if the path segment of the shape crosses the line clockwise, one is added, otherwise one is subtracted. If the resulting sum is non zero, the point is inside the shape.
    EvenOdd      ///< A line from the point to a location outside the shape is drawn and its intersections with the path segments of the shape are counted. If the number of intersections is an odd number, the point is inside the shape.
};

/**
 * @brief Enumeration indicating the method used in the composition of two objects - the target and the source.
 */
enum class CompositeMethod
{
    None = 0,     ///< No composition is applied.
    ClipPath,     ///< The intersection of the source and the target is determined and only the resulting pixels from the source are rendered.
    AlphaMask,    ///< The pixels of the source and the target are alpha blended. As a result, only the part of the source, which alpha intersects with the target is visible.
    InvAlphaMask, ///< The pixels of the source and the complement to the target's pixels are alpha blended. As a result, only the part of the source which alpha is not covered by the target is visible.
    LumaMask      ///< @BETA_API The source pixels are converted to the grayscale (luma value) and alpha blended with the target. As a result, only the part of the source, which intersects with the target is visible.
};

/**
 * @brief Enumeration specifying the engine type used for the graphics backend. For multiple backends bitwise operation is allowed.
 */
enum class CanvasEngine
{
    Sw = (1 << 1), ///< CPU rasterizer.
    Gl = (1 << 2)  ///< OpenGL rasterizer.
};


/**
 * @brief A data structure representing a point in two-dimensional space.
 */
struct Point
{
    float x, y;
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
 * @class Paint
 *
 * @brief An abstract class for managing graphical elements.
 *
 * A graphical element in TVG is any object composed into a Canvas.
 * Paint represents such a graphical object and its behaviors such as duplication, transformation and composition.
 * TVG recommends the user to regard a paint as a set of volatile commands. They can prepare a Paint and then request a Canvas to run them.
 */
class TVG_EXPORT Paint
{
public:
    virtual ~Paint();

    /**
     * @brief Sets the angle by which the object is rotated.
     *
     * The angle in measured clockwise from the horizontal axis.
     * The rotational axis passes through the point on the object with zero coordinates.
     *
     * @param[in] degree The value of the angle in degrees.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result rotate(float degree) noexcept;

    /**
     * @brief Sets the scale value of the object.
     *
     * @param[in] factor The value of the scaling factor. The default value is 1.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result scale(float factor) noexcept;

    /**
     * @brief Sets the values by which the object is moved in a two-dimensional space.
     *
     * The origin of the coordinate system is in the upper left corner of the canvas.
     * The horizontal and vertical axes point to the right and down, respectively.
     *
     * @param[in] x The value of the horizontal shift.
     * @param[in] y The value of the vertical shift.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result translate(float x, float y) noexcept;

    /**
     * @brief Sets the matrix of the affine transformation for the object.
     *
     * The augmented matrix of the transformation is expected to be given.
     *
     * @param[in] m The 3x3 augmented matrix.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result transform(const Matrix& m) noexcept;

    /**
     * @brief Gets the matrix of the affine transformation of the object.
     *
     * The values of the matrix can be set by the transform() API, as well by the translate(),
     * scale() and rotate(). In case no transformation was applied, the identity matrix is returned.
     *
     * @retval The augmented transformation matrix.
     *
     * @since 0.4
     */
    Matrix transform() noexcept;

    /**
     * @brief Sets the opacity of the object.
     *
     * @param[in] o The opacity value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque.
     *
     * @return Result::Success when succeed.
     *
     * @note Setting the opacity with this API may require multiple render pass for composition. It is recommended to avoid changing the opacity if possible.
     */
    Result opacity(uint8_t o) noexcept;

    /**
     * @brief Sets the composition target object and the composition method.
     *
     * @param[in] target The paint of the target object.
     * @param[in] method The method used to composite the source object with the target.
     *
     * @return Result::Success when succeed, Result::InvalidArguments otherwise.
     */
    Result composite(std::unique_ptr<Paint> target, CompositeMethod method) noexcept;

    /**
     * @brief Gets the bounding box of the paint object before any transformation.
     *
     * @param[out] x The x coordinate of the upper left corner of the object.
     * @param[out] y The y coordinate of the upper left corner of the object.
     * @param[out] w The width of the object.
     * @param[out] h The height of the object.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     *
     * @note The bounding box doesn't indicate the final rendered region. It's the smallest rectangle that encloses the object.
     * @see Paint::bounds(float* x, float* y, float* w, float* h, bool transformed);
     */
    TVG_DEPRECATED Result bounds(float* x, float* y, float* w, float* h) const noexcept;

    /**
     * @brief Gets the axis-aligned bounding box of the paint object.
     *
     * In case @p transform is @c true, all object's transformations are applied first, and then the bounding box is established. Otherwise, the bounding box is determined before any transformations.
     *
     * @param[out] x The x coordinate of the upper left corner of the object.
     * @param[out] y The y coordinate of the upper left corner of the object.
     * @param[out] w The width of the object.
     * @param[out] h The height of the object.
     * @param[in] transformed If @c true, the paint's transformations are taken into account, otherwise they aren't.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     *
     * @note The bounding box doesn't indicate the actual drawing region. It's the smallest rectangle that encloses the object.
     */
    Result bounds(float* x, float* y, float* w, float* h, bool transformed) const noexcept;

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
     * @brief Gets the composition target object and the composition method.
     *
     * @param[out] target The paint of the target object.
     *
     * @return The method used to composite the source object with the target.
     *
     * @since 0.5
     */
    CompositeMethod composite(const Paint** target) const noexcept;

    /**
     * @brief Return the unique id value of the paint instance.
     *
     * This method can be called for checking the current concrete instance type.
     *
     * @return The type id of the Paint instance.
     *
     * @BETA_API
     */
    uint32_t identifier() const noexcept;

    _TVG_DECLARE_ACCESSOR();
    _TVG_DECLARE_PRIVATE(Paint);
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
class TVG_EXPORT Fill
{
public:
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
     *
     * @return Result::Success when succeed.
     */
    Result colorStops(const ColorStop* colorStops, uint32_t cnt) noexcept;

    /**
     * @brief Sets the FillSpread value, which specifies how to fill the area outside the gradient bounds.
     *
     * @param[in] s The FillSpread value.
     *
     * @return Result::Success when succeed.
     */
    Result spread(FillSpread s) noexcept;

    /**
     * @brief Sets the matrix of the affine transformation for the gradient fill.
     *
     * The augmented matrix of the transformation is expected to be given.
     *
     * @param[in] m The 3x3 augmented matrix.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
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
     * @retval The augmented transformation matrix.
     */
    Matrix transform() const noexcept;

    /**
     * @brief Creates a copy of the Fill object.
     *
     * Return a newly created Fill object with the properties copied from the original.
     *
     * @return A copied Fill object when succeed, @c nullptr otherwise.
     */
    Fill* duplicate() const noexcept;

    /**
     * @brief Return the unique id value of the Fill instance.
     *
     * This method can be called for checking the current concrete instance type.
     *
     * @return The type id of the Fill instance.
     *
     * @BETA_API
     */
    uint32_t identifier() const noexcept;

    _TVG_DECLARE_PRIVATE(Fill);
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
class TVG_EXPORT Canvas
{
public:
    Canvas(RenderMethod*);
    virtual ~Canvas();

    /**
     * @brief Sets the size of the container, where all the paints pushed into the Canvas are stored.
     *
     * If the number of objects pushed into the Canvas is known in advance, calling the function
     * prevents multiple memory reallocation, thus improving the performance.
     *
     * @param[in] n The number of objects for which the memory is to be reserved.
     *
     * @return Result::Success when succeed.
     */
    Result reserve(uint32_t n) noexcept;

    /**
     * @brief Passes drawing elements to the Canvas using Paint objects.
     *
     * Only pushed paints in the canvas will be drawing targets.
     * They are retained by the canvas until you call Canvas::clear().
     * If you know the number of the pushed objects in advance, please call Canvas::reserve().
     *
     * @param[in] paint A Paint object to be drawn.
     *
     * @retval Result::Success When succeed.
     * @retval Result::MemoryCorruption In case a @c nullptr is passed as the argument.
     * @retval Result::InsufficientCondition An internal error.
     *
     * @note The rendering order of the paints is the same as the order as they were pushed into the canvas. Consider sorting the paints before pushing them if you intend to use layering.
     * @see Canvas::reserve()
     * @see Canvas::clear()
     */
    virtual Result push(std::unique_ptr<Paint> paint) noexcept;

    /**
     * @brief Sets the total number of the paints pushed into the canvas to be zero.
     * Depending on the value of the @p free argument, the paints are freed or not.
     *
     * @param[in] free If @c true, the memory occupied by paints is deallocated, otherwise it is not.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     *
     * @warning If you don't free the paints they become dangled. They are supposed to be reused, otherwise you are responsible for their lives. Thus please use the @p free argument only when you know how it works, otherwise it's not recommended.
     */
    virtual Result clear(bool free = true) noexcept;

    /**
     * @brief Request the canvas to update the paint objects.
     *
     * If a @c nullptr is passed all paint objects retained by the Canvas are updated,
     * otherwise only the paint to which the given @p paint points.
     *
     * @param[in] paint A pointer to the Paint object or @c nullptr.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     *
     * @note The Update behavior can be asynchronous if the assigned thread number is greater than zero.
     */
    virtual Result update(Paint* paint = nullptr) noexcept;

    /**
     * @brief Requests the canvas to draw the Paint objects.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     *
     * @note Drawing can be asynchronous if the assigned thread number is greater than zero. To guarantee the drawing is done, call sync() afterwards.
     * @see Canvas::sync()
     */
    virtual Result draw() noexcept;

    /**
     * @brief Guarantees that drawing task is finished.
     *
     * The Canvas rendering can be performed asynchronously. To make sure that rendering is finished,
     * the sync() must be called after the draw() regardless of threading.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     * @see Canvas::draw()
     */
    virtual Result sync() noexcept;

    _TVG_DECLARE_PRIVATE(Canvas);
};


/**
 * @class LinearGradient
 *
 * @brief A class representing the linear gradient fill of the Shape object.
 *
 * Besides the APIs inherited from the Fill class, it enables setting and getting the linear gradient bounds.
 * The behavior outside the gradient bounds depends on the value specified in the spread API.
 */
class TVG_EXPORT LinearGradient final : public Fill
{
public:
    ~LinearGradient();

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
     * @return Result::Success when succeed.
     *
     * @note In case the first and the second points are equal, an object filled with such a gradient fill is not rendered.
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
     *
     * @return Result::Success when succeed.
     */
    Result linear(float* x1, float* y1, float* x2, float* y2) const noexcept;

    /**
     * @brief Creates a new LinearGradient object.
     *
     * @return A new LinearGradient object.
     */
    static std::unique_ptr<LinearGradient> gen() noexcept;

    /**
     * @brief Return the unique id value of this class.
     *
     * This method can be referred for identifying the LinearGradient class type.
     *
     * @return The type id of the LinearGradient class.
     *
     * @BETA_API
     */
    static uint32_t identifier() noexcept;

    _TVG_DECLARE_PRIVATE(LinearGradient);
};


/**
 * @class RadialGradient
 *
 * @brief A class representing the radial gradient fill of the Shape object.
 *
 */
class TVG_EXPORT RadialGradient final : public Fill
{
public:
    ~RadialGradient();

    /**
     * @brief Sets the radial gradient bounds.
     *
     * The radial gradient bounds are defined as a circle centered in a given point (@p cx, @p cy) of a given radius.
     *
     * @param[in] cx The horizontal coordinate of the center of the bounding circle.
     * @param[in] cy The vertical coordinate of the center of the bounding circle.
     * @param[in] radius The radius of the bounding circle.
     *
     * @return Result::Success when succeed, Result::InvalidArguments in case the @p radius value is zero or less.
     */
    Result radial(float cx, float cy, float radius) noexcept;

    /**
     * @brief Gets the radial gradient bounds.
     *
     * The radial gradient bounds are defined as a circle centered in a given point (@p cx, @p cy) of a given radius.
     *
     * @param[out] cx The horizontal coordinate of the center of the bounding circle.
     * @param[out] cy The vertical coordinate of the center of the bounding circle.
     * @param[out] radius The radius of the bounding circle.
     *
     * @return Result::Success when succeed.
     */
    Result radial(float* cx, float* cy, float* radius) const noexcept;

    /**
     * @brief Creates a new RadialGradient object.
     *
     * @return A new RadialGradient object.
     */
    static std::unique_ptr<RadialGradient> gen() noexcept;

    /**
     * @brief Return the unique id value of this class.
     *
     * This method can be referred for identifying the RadialGradient class type.
     *
     * @return The type id of the RadialGradient class.
     *
     * @BETA_API
     */
    static uint32_t identifier() noexcept;

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
 */
class TVG_EXPORT Shape final : public Paint
{
public:
    ~Shape();

    /**
     * @brief Resets the properties of the shape path.
     *
     * The color, the fill and the stroke properties are retained.
     *
     * @return Result::Success when succeed.
     *
     * @note The memory, where the path data is stored, is not deallocated at this stage for caching effect.
     */
    Result reset() noexcept;

    /**
     * @brief Sets the initial point of the sub-path.
     *
     * The value of the current point is set to the given point.
     *
     * @param[in] x The horizontal coordinate of the initial point of the sub-path.
     * @param[in] y The vertical coordinate of the initial point of the sub-path.
     *
     * @return Result::Success when succeed.
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
     * @return Result::Success when succeed.
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
     * @return Result::Success when succeed.
     *
     * @note In case this is the first command in the path, no data from the path are rendered.
     */
    Result cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y) noexcept;

    /**
     * @brief Closes the current sub-path by drawing a line from the current point to the initial point of the sub-path.
     *
     * The value of the current point is set to the initial point of the closed sub-path.
     *
     * @return Result::Success when succeed.
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
     * The position of the rectangle is specified by the coordinates of its upper left corner - @p x and @p y arguments.
     *
     * The rectangle is treated as a new sub-path - it is not connected with the previous sub-path.
     *
     * The value of the current point is set to (@p x + @p rx, @p y) - in case @p rx is greater
     * than @p w/2 the current point is set to (@p x + @p w/2, @p y)
     *
     * @param[in] x The horizontal coordinate of the upper left corner of the rectangle.
     * @param[in] y The vertical coordinate of the upper left corner of the rectangle.
     * @param[in] w The width of the rectangle.
     * @param[in] h The height of the rectangle.
     * @param[in] rx The x-axis radius of the ellipse defining the rounded corners of the rectangle.
     * @param[in] ry The y-axis radius of the ellipse defining the rounded corners of the rectangle.
     *
     * @return Result::Success when succeed.
     *
     * @note For @p rx and @p ry greater than or equal to the half of @p w and the half of @p h, respectively, the shape become an ellipse.
     */
    Result appendRect(float x, float y, float w, float h, float rx, float ry) noexcept;

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
     *
     * @return Result::Success when succeed.
     */
    Result appendCircle(float cx, float cy, float rx, float ry) noexcept;

    /**
     * @brief Appends a circular arc to the path.
     *
     * The arc is treated as a new sub-path - it is not connected with the previous sub-path.
     * The current point value is set to the end-point of the arc in case @p pie is @c false, and to the center of the arc otherwise.
     *
     * @param[in] cx The horizontal coordinate of the center of the arc.
     * @param[in] cy The vertical coordinate of the center of the arc.
     * @param[in] radius The radius of the arc.
     * @param[in] startAngle The start angle of the arc given in degrees, measured counter-clockwise from the horizontal line.
     * @param[in] sweep The central angle of the arc given in degrees, measured counter-clockwise from @p startAngle.
     * @param[in] pie Specifies whether to draw radii from the arc's center to both of its end-point - drawn if @c true.
     *
     * @return Result::Success when succeed.
     *
     * @note Setting @p sweep value greater than 360 degrees, is equivalent to calling appendCircle(cx, cy, radius, radius).
     */
    Result appendArc(float cx, float cy, float radius, float startAngle, float sweep, bool pie) noexcept;

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
     * @return Result::Success when succeed, Result::InvalidArguments otherwise.
     *
     * @note The interface is designed for optimal path setting if the caller has a completed path commands already.
     */
    Result appendPath(const PathCommand* cmds, uint32_t cmdCnt, const Point* pts, uint32_t ptsCnt) noexcept;

    /**
     * @brief Sets the stroke width for all of the figures from the path.
     *
     * @param[in] width The width of the stroke. The default value is 0.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result stroke(float width) noexcept;

    /**
     * @brief Sets the color of the stroke for all of the figures from the path.
     *
     * @param[in] r The red color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] g The green color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] b The blue color channel value in the range [0 ~ 255]. The default value is 0.
     * @param[in] a The alpha channel value in the range [0 ~ 255], where 0 is completely transparent and 255 is opaque. The default value is 0.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result stroke(uint8_t r, uint8_t g, uint8_t b, uint8_t a) noexcept;

    /**
     * @brief Sets the gradient fill of the stroke for all of the figures from the path.
     *
     * @param[in] f The gradient fill.
     *
     * @retval Result::Success When succeed.
     * @retval Result::FailedAllocation An internal error with a memory allocation for an object to be filled.
     * @retval Result::MemoryCorruption In case a @c nullptr is passed as the argument.
     */
    Result stroke(std::unique_ptr<Fill> f) noexcept;

    /**
     * @brief Sets the dash pattern of the stroke.
     *
     * @param[in] dashPattern The array of consecutive pair values of the dash length and the gap length.
     * @param[in] cnt The length of the @p dashPattern array.
     *
     * @retval Result::Success When succeed.
     * @retval Result::FailedAllocation An internal error with a memory allocation for an object to be dashed.
     * @retval Result::InvalidArguments In case @p dashPattern is @c nullptr and @p cnt > 0, @p cnt is zero, any of the dash pattern values is zero or less.
     *
     * @note To reset the stroke dash pattern, pass @c nullptr to @p dashPattern and zero to @p cnt.
     * @warning @p cnt must be greater than 1 if the dash pattern is valid.
     */
    Result stroke(const float* dashPattern, uint32_t cnt) noexcept;

    /**
     * @brief Sets the cap style of the stroke in the open sub-paths.
     *
     * @param[in] cap The cap style value. The default value is @c StrokeCap::Square.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result stroke(StrokeCap cap) noexcept;

    /**
     * @brief Sets the join style for stroked path segments.
     *
     * The join style is used for joining the two line segment while stroking the path.
     *
     * @param[in] join The join style value. The default value is @c StrokeJoin::Bevel.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result stroke(StrokeJoin join) noexcept;

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
     * @return Result::Success when succeed.
     *
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     */
    Result fill(uint8_t r, uint8_t g, uint8_t b, uint8_t a) noexcept;

    /**
     * @brief Sets the gradient fill for all of the figures from the path.
     *
     * The parts of the shape defined as inner are filled.
     *
     * @param[in] f The unique pointer to the gradient fill.
     *
     * @return Result::Success when succeed, Result::MemoryCorruption otherwise.
     *
     * @note Either a solid color or a gradient fill is applied, depending on what was set as last.
     */
    Result fill(std::unique_ptr<Fill> f) noexcept;

    /**
     * @brief Sets the fill rule for the Shape object.
     *
     * @param[in] r The fill rule value. The default value is @c FillRule::Winding.
     *
     * @return Result::Success when succeed.
     */
    Result fill(FillRule r) noexcept;

    /**
     * @brief Gets the commands data of the path.
     *
     * @param[out] cmds The pointer to the array of the commands from the path.
     *
     * @return The length of the @p cmds array when succeed, zero otherwise.
     */
    uint32_t pathCommands(const PathCommand** cmds) const noexcept;

    /**
     * @brief Gets the points values of the path.
     *
     * @param[out] pts The pointer to the array of the two-dimensional points from the path.
     *
     * @return The length of the @p pts array when succeed, zero otherwise.
     */
    uint32_t pathCoords(const Point** pts) const noexcept;

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
     * @return Result::Success when succeed.
     */
    Result fillColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const noexcept;

    /**
     * @brief Gets the fill rule value.
     *
     * @return The fill rule value of the shape.
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
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     */
    Result strokeColor(uint8_t* r, uint8_t* g, uint8_t* b, uint8_t* a) const noexcept;

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
     *
     * @return The length of the @p dashPattern array.
     */
    uint32_t strokeDash(const float** dashPattern) const noexcept;

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
     * @brief Creates a new Shape object.
     *
     * @return A new Shape object.
     */
    static std::unique_ptr<Shape> gen() noexcept;

    /**
     * @brief Return the unique id value of this class.
     *
     * This method can be referred for identifying the Shape class type.
     *
     * @return The type id of the Shape class.
     *
     * @BETA_API
     */
    static uint32_t identifier() noexcept;

    _TVG_DECLARE_PRIVATE(Shape);
};


/**
 * @class Picture
 *
 * @brief A class representing an image read in one of the supported formats: raw, svg, png, jpg and etc.
 * Besides the methods inherited from the Paint, it provides methods to load & draw images on the canvas.
 *
 * @note Supported formats are depended on the available TVG loaders.
 */
class TVG_EXPORT Picture final : public Paint
{
public:
    ~Picture();

    /**
     * @brief Loads a picture data directly from a file.
     *
     * @param[in] path A path to the picture file.
     *
     * @retval Result::Success When succeed.
     * @retval Result::InvalidArguments In case the @p path is invalid.
     * @retval Result::NonSupport When trying to load a file with an unknown extension.
     * @retval Result::Unknown If an error occurs at a later stage.
     *
     * @note The Load behavior can be asynchronous if the assigned thread number is greater than zero.
     * @see Initializer::init()
     */
    Result load(const std::string& path) noexcept;

    /**
     * @brief Loads a picture data from a memory block of a given size.
     *
     * @param[in] data A pointer to a memory location where the content of the picture file is stored.
     * @param[in] size The size in bytes of the memory occupied by the @p data.
     * @param[in] copy Decides whether the data should be copied into the engine local buffer.
     *
     * @retval Result::Success When succeed.
     * @retval Result::InvalidArguments In case no data are provided or the @p size is zero or less.
     * @retval Result::NonSupport When trying to load a file with an unknown extension.
     * @retval Result::Unknown If an error occurs at a later stage.
     *
     * @warning: you have responsibility to release the @p data memory if the @p copy is true
     * @deprecated Use load(const char* data, uint32_t size, const std::string& mimeType, bool copy) instead.
     * @see Result load(const char* data, uint32_t size, const std::string& mimeType, bool copy = false) noexcept
     */
    TVG_DEPRECATED Result load(const char* data, uint32_t size, bool copy = false) noexcept;

    /**
     * @brief Loads a picture data from a memory block of a given size.
     *
     * @param[in] data A pointer to a memory location where the content of the picture file is stored.
     * @param[in] size The size in bytes of the memory occupied by the @p data.
     * @param[in] mimeType Mimetype or extension of data such as "jpg", "jpeg", "svg", "svg+xml", "png", etc. In case an empty string or an unknown type is provided, the loaders will be tried one by one.
     * @param[in] copy If @c true the data are copied into the engine local buffer, otherwise they are not.
     *
     * @retval Result::Success When succeed.
     * @retval Result::InvalidArguments In case no data are provided or the @p size is zero or less.
     * @retval Result::NonSupport When trying to load a file with an unknown extension.
     * @retval Result::Unknown If an error occurs at a later stage.
     *
     * @warning: It's the user responsibility to release the @p data memory if the @p copy is @c true.
     *
     * @since 0.5
     */
    Result load(const char* data, uint32_t size, const std::string& mimeType, bool copy = false) noexcept;

    /**
     * @brief Resizes the picture content to the given width and height.
     *
     * The picture content is resized while keeping the default size aspect ratio.
     * The scaling factor is established for each of dimensions and the smaller value is applied to both of them.
     *
     * @param[in] w A new width of the image in pixels.
     * @param[in] h A new height of the image in pixels.
     *
     * @return Result::Success when succeed, Result::InsufficientCondition otherwise.
     */
    Result size(float w, float h) noexcept;

    /**
     * @brief Gets the size of the image.
     *
     * @param[out] w The width of the image in pixels.
     * @param[out] h The height of the image in pixels.
     *
     * @return Result::Success when succeed.
     */
    Result size(float* w, float* h) const noexcept;

    /**
     * @brief Gets the pixels information of the picture.
     *
     * @note The data must be pre-multiplied by the alpha channels.
     *
     * @warning Please do not use it, this API is not official one. It could be modified in the next version.
     *
     * @BETA_API
     */
    const uint32_t* data(uint32_t* w, uint32_t* h) const noexcept;

    /**
     * @brief Loads a raw data from a memory block with a given size.
     *
     * @warning Please do not use it, this API is not official one. It could be modified in the next version.
     *
     * @BETA_API
     */
    Result load(uint32_t* data, uint32_t w, uint32_t h, bool copy) noexcept;

    /**
     * @brief Gets the position and the size of the loaded SVG picture.
     *
     * @warning Please do not use it, this API is not official one. It could be modified in the next version.
     *
     * @BETA_API
     */
    Result viewbox(float* x, float* y, float* w, float* h) const noexcept;

    /**
     * @brief Creates a new Picture object.
     *
     * @return A new Picture object.
     */
    static std::unique_ptr<Picture> gen() noexcept;

    /**
     * @brief Return the unique id value of this class.
     *
     * This method can be referred for identifying the Picture class type.
     *
     * @return The type id of the Picture class.
     *
     * @BETA_API
     */
    static uint32_t identifier() noexcept;

    _TVG_DECLARE_PRIVATE(Picture);
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
 */
class TVG_EXPORT Scene final : public Paint
{
public:
    ~Scene();

    /**
     * @brief Passes drawing elements to the Scene using Paint objects.
     *
     * Only the paints pushed into the scene will be the drawn targets.
     * The paints are retained by the scene until Scene::clear() is called.
     * If you know the number of the pushed objects in advance, please call Scene::reserve().
     *
     * @param[in] paint A Paint object to be drawn.
     *
     * @return Result::Success when succeed, Result::MemoryCorruption otherwise.
     *
     * @note The rendering order of the paints is the same as the order as they were pushed. Consider sorting the paints before pushing them if you intend to use layering.
     * @see Scene::reserve()
     */
    Result push(std::unique_ptr<Paint> paint) noexcept;

    /**
     * @brief Sets the size of the container, where all the paints pushed into the Scene are stored.
     *
     * If the number of objects pushed into the scene is known in advance, calling the function
     * prevents multiple memory reallocation, thus improving the performance.
     *
     * @param[in] size The number of objects for which the memory is to be reserved.
     *
     * @return Result::Success when succeed, Result::FailedAllocation otherwise.
     */
    Result reserve(uint32_t size) noexcept;

    /**
     * @brief Sets the total number of the paints pushed into the scene to be zero.
     * Depending on the value of the @p free argument, the paints are freed or not.
     *
     * @param[in] free If @c true, the memory occupied by paints is deallocated, otherwise it is not.
     *
     * @return Result::Success when succeed
     *
     * @warning If you don't free the paints they become dangled. They are supposed to be reused, otherwise you are responsible for their lives. Thus please use the @p free argument only when you know how it works, otherwise it's not recommended.
     *
     * @since 0.2
     */
    Result clear(bool free = true) noexcept;

    /**
     * @brief Creates a new Scene object.
     *
     * @return A new Scene object.
     */
    static std::unique_ptr<Scene> gen() noexcept;

    /**
     * @brief Return the unique id value of this class.
     *
     * This method can be referred for identifying the Scene class type.
     *
     * @return The type id of the Scene class.
     *
     * @BETA_API
     */
    static uint32_t identifier() noexcept;

    _TVG_DECLARE_PRIVATE(Scene);
};


/**
 * @class SwCanvas
 *
 * @brief A class for the rendering graphical elements with a software raster engine.
 */
class TVG_EXPORT SwCanvas final : public Canvas
{
public:
    ~SwCanvas();

    /**
     * @brief Enumeration specifying the methods of combining the 8-bit color channels into 32-bit color.
     */
    enum Colorspace
    {
        ABGR8888 = 0,      ///< The channels are joined in the order: alpha, blue, green, red. Colors are alpha-premultiplied.
        ARGB8888,          ///< The channels are joined in the order: alpha, red, green, blue. Colors are alpha-premultiplied.
        ABGR8888_STRAIGHT, ///< @BETA_API The channels are joined in the order: alpha, blue, green, red. Colors are un-alpha-premultiplied.
        ARGB8888_STRAIGHT, ///< @BETA_API The channels are joined in the order: alpha, red, green, blue. Colors are un-alpha-premultiplied.
    };

    /**
     * @brief Enumeration specifying the methods of Memory Pool behavior policy.
     * @since 0.4
     */
    enum MempoolPolicy
    {
        Default = 0, ///< Default behavior that ThorVG is designed to.
        Shareable,   ///< Memory Pool is shared among the SwCanvases.
        Individual   ///< Allocate designated memory pool that is only used by current instance.
    };

    /**
     * @brief Sets the target buffer for the rasterization.
     *
     * The buffer of a desirable size should be allocated and owned by the caller.
     *
     * @param[in] buffer A pointer to a memory block of the size @p stride x @p h, where the raster data are stored.
     * @param[in] stride The stride of the raster image - greater than or equal to @p w.
     * @param[in] w The width of the raster image.
     * @param[in] h The height of the raster image.
     * @param[in] cs The value specifying the way the 32-bits colors should be read/written.
     *
     * @retval Result::Success When succeed.
     * @retval Result::MemoryCorruption When casting in the internal function implementation failed.
     * @retval Result::InvalidArguments In case no valid pointer is provided or the width, or the height or the stride is zero.
     * @retval Result::NonSupport In case the software engine is not supported.
     *
     * @warning Do not access @p buffer during Canvas::draw() - Canvas::sync(). It should not be accessed while TVG is writing on it.
    */
    Result target(uint32_t* buffer, uint32_t stride, uint32_t w, uint32_t h, Colorspace cs) noexcept;

    /**
     * @brief Set sw engine memory pool behavior policy.
     *
     * Basically ThorVG draws a lot of shapes, it allocates/deallocates a few chunk of memory
     * while processing rendering. It internally uses one shared memory pool
     * which can be reused among the canvases in order to avoid memory overhead.
     *
     * Thus ThorVG suggests using a memory pool policy to satisfy user demands,
     * if it needs to guarantee the thread-safety of the internal data access.
     *
     * @param[in] policy The method specifying the Memory Pool behavior. The default value is @c MempoolPolicy::Default.
     *
     * @retval Result::Success When succeed.
     * @retval Result::InsufficientCondition If the canvas contains some paints already.
     * @retval Result::NonSupport In case the software engine is not supported.
     *
     * @note When @c policy is set as @c MempoolPolicy::Individual, the current instance of canvas uses its own individual
     *       memory data, which is not shared with others. This is necessary when the canvas is accessed on a worker-thread.
     *
     * @warning It's not allowed after pushing any paints.
     *
     * @since 0.4
    */
    Result mempool(MempoolPolicy policy) noexcept;

    /**
     * @brief Creates a new SwCanvas object.
     * @return A new SwCanvas object.
     */
    static std::unique_ptr<SwCanvas> gen() noexcept;

    _TVG_DECLARE_PRIVATE(SwCanvas);
};


/**
 * @class GlCanvas
 *
 * @brief A class for the rendering graphic elements with a GL raster engine.
 *
 * @warning Please do not use it. This class is not fully supported yet.
 *
 * @BETA_API
 */
class TVG_EXPORT GlCanvas final : public Canvas
{
public:
    ~GlCanvas();

    /**
     * @brief Sets the target buffer for the rasterization.
     *
     * @warning Please do not use it, this API is not official one. It could be modified in the next version.
     *
     * @BETA_API
     */
    Result target(uint32_t* buffer, uint32_t stride, uint32_t w, uint32_t h) noexcept;

    /**
     * @brief Creates a new GlCanvas object.
     *
     * @return A new GlCanvas object.
     *
     * @BETA_API
     */
    static std::unique_ptr<GlCanvas> gen() noexcept;

    _TVG_DECLARE_PRIVATE(GlCanvas);
};


/**
 * @class Initializer
 *
 * @brief A class that enables initialization and termination of the TVG engines.
 */
class TVG_EXPORT Initializer final
{
public:
    /**
     * @brief Initializes TVG engines.
     *
     * TVG requires the running-engine environment.
     * TVG runs its own task-scheduler for parallelizing rendering tasks efficiently.
     * You can indicate the number of threads, the count of which is designated @p threads.
     * In the initialization step, TVG will generate/spawn the threads as set by @p threads count.
     *
     * @param[in] engine The engine types to initialize. This is relative to the Canvas types, in which it will be used. For multiple backends bitwise operation is allowed.
     * @param[in] threads The number of additional threads. Zero indicates only the main thread is to be used.
     *
     * @retval Result::Success When succeed.
     * @retval Result::FailedAllocation An internal error possibly with memory allocation.
     * @retval Result::InvalidArguments If unknown engine type chosen.
     * @retval Result::NonSupport In case the engine type is not supported on the system.
     * @retval Result::Unknown Others.
     *
     * @note The Initializer keeps track of the number of times it was called. Threads count is fixed at the first init() call.
     * @see Initializer::term()
     */
    static Result init(CanvasEngine engine, uint32_t threads) noexcept;

    /**
     * @brief Terminates TVG engines.
     *
     * @param[in] engine The engine types to terminate. This is relative to the Canvas types, in which it will be used. For multiple backends bitwise operation is allowed
     *
     * @retval Result::Success When succeed.
     * @retval Result::InsufficientCondition In case there is nothing to be terminated.
     * @retval Result::InvalidArguments If unknown engine type chosen.
     * @retval Result::NonSupport In case the engine type is not supported on the system.
     * @retval Result::Unknown Others.
     *
     * @note Initializer does own reference counting for multiple calls.
     * @see Initializer::init()
     */
    static Result term(CanvasEngine engine) noexcept;

    _TVG_DISABLE_CTOR(Initializer);
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
 * If it doesn't support the file format, the save() method returns the @c Result::NonSuppport result.
 *
 * Once you export a paint to the file successfully, you can recreate it using the Picture class.
 *
 * @see Picture::load()
 *
 * @since 0.5
 */
class TVG_EXPORT Saver final
{
public:
    ~Saver();

    /**
     * @brief Exports the given @p paint data to the given @p path
     *
     * If the saver module supports any compression mechanism, it will optimize the data size.
     * This might affect the encoding/decoding time in some cases. You can turn off the compression
     * if you wish to optimize for speed.
     *
     * @param[in] paint The paint to be saved with all its associated properties.
     * @param[in] path A path to the file, in which the paint data is to be saved.
     * @param[in] compress If @c true then compress data if possible.
     *
     * @retval Result::Success When succeed.
     * @retval Result::InsufficientCondition If currently saving other resources.
     * @retval Result::NonSupport When trying to save a file with an unknown extension or in an unsupported format.
     * @retval Result::MemoryCorruption An internal error.
     * @retval Result::Unknown In case an empty paint is to be saved.
     *
     * @note Saving can be asynchronous if the assigned thread number is greater than zero. To guarantee the saving is done, call sync() afterwards.
     * @see Saver::sync()
     *
     * @since 0.5
     */
    Result save(std::unique_ptr<Paint> paint, const std::string& path, bool compress = true) noexcept;

    /**
     * @brief Guarantees that the saving task is finished.
     *
     * The behavior of the Saver works on a sync/async basis, depending on the threading setting of the Initializer.
     * Thus, if you wish to have a benefit of it, you must call sync() after the save() in the proper delayed time.
     * Otherwise, you can call sync() immediately.
     *
     * @retval Result::Success when succeed.
     * @retval Result::InsufficientCondition otherwise.
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
    static std::unique_ptr<Saver> gen() noexcept;

    _TVG_DECLARE_PRIVATE(Saver);
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
 * @BETA_API
 */
class TVG_EXPORT Accessor final
{
public:
    ~Accessor();

    /**
     * @brief Access the Picture scene stree nodes.
     *
     * @param[in] picture The picture node to traverse the internal scene-tree.
     * @param[in] func The callback function calling for every paint nodes of the Picture.
     *
     * @return Return the given @p picture instance.
     *
     * @note The bitmap based picture might not have the scene-tree.
     *
     * @BETA_API
     */
    std::unique_ptr<Picture> access(std::unique_ptr<Picture> picture, bool(*func)(const Paint* paint)) noexcept;

    /**
     * @brief Creates a new Accessor object.
     *
     * @return A new Accessor object.
     *
     * @BETA_API
     */
    static std::unique_ptr<Accessor> gen() noexcept;

    _TVG_DECLARE_PRIVATE(Accessor);
};

/** @}*/

} //namespace

#ifdef __cplusplus
}
#endif

#endif //_THORVG_H_
