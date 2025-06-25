/*
 * Copyright (c) 2020-2024 Samuel Ugochukwu <sammycageagle@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
*/

#ifndef LUNASVG_H
#define LUNASVG_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#if !defined(LUNASVG_BUILD_STATIC) && (defined(_WIN32) || defined(__CYGWIN__))
#define LUNASVG_EXPORT __declspec(dllexport)
#define LUNASVG_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#define LUNASVG_EXPORT __attribute__((__visibility__("default")))
#define LUNASVG_IMPORT
#else
#define LUNASVG_EXPORT
#define LUNASVG_IMPORT
#endif

#ifdef LUNASVG_BUILD
#define LUNASVG_API LUNASVG_EXPORT
#else
#define LUNASVG_API LUNASVG_IMPORT
#endif

#define LUNASVG_VERSION_MAJOR 3
#define LUNASVG_VERSION_MINOR 1
#define LUNASVG_VERSION_MICRO 0

#define LUNASVG_VERSION_ENCODE(major, minor, micro) (((major) * 10000) + ((minor) * 100) + ((micro) * 1))
#define LUNASVG_VERSION LUNASVG_VERSION_ENCODE(LUNASVG_VERSION_MAJOR, LUNASVG_VERSION_MINOR, LUNASVG_VERSION_MICRO)

#define LUNASVG_VERSION_XSTRINGIZE(major, minor, micro) #major"."#minor"."#micro
#define LUNASVG_VERSION_STRINGIZE(major, minor, micro) LUNASVG_VERSION_XSTRINGIZE(major, minor, micro)
#define LUNASVG_VERSION_STRING LUNASVG_VERSION_STRINGIZE(LUNASVG_VERSION_MAJOR, LUNASVG_VERSION_MINOR, LUNASVG_VERSION_MICRO)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct plutovg_surface plutovg_surface_t;
typedef struct plutovg_matrix plutovg_matrix_t;

/**
 * @brief Callback for cleaning up resources.
 *
 * This function is called to release resources associated with a specific operation.
 *
 * @param closure A user-defined pointer to the resource or context to be freed.
 */
typedef void (*lunasvg_destroy_func_t)(void* closure);

/**
 * @brief A function pointer type for a write callback.
 * @param closure A pointer to user-defined data or context.
 * @param data A pointer to the data to be written.
 * @param size The size of the data in bytes.
 */
typedef void (*lunasvg_write_func_t)(void* closure, void* data, int size);

/**
 * @brief Returns the version of the lunasvg library encoded in a single integer.
 *
 * Encodes the version of the lunasvg library into a single integer for easier comparison.
 * The version is typically represented by combining major, minor, and patch numbers into one integer.
 *
 * @return The lunasvg library version as a single integer.
 */
LUNASVG_API int lunasvg_version(void);

/**
 * @brief Returns the lunasvg library version as a human-readable string in "X.Y.Z" format.
 *
 * Provides the version of the lunasvg library as a human-readable string in the format "X.Y.Z",
 * where X represents the major version, Y the minor version, and Z the patch version.
 *
 * @return A pointer to a string containing the version in "X.Y.Z" format.
 */
LUNASVG_API const char* lunasvg_version_string(void);

/**
* @brief Add a font face from a file to the cache.
* @param family The name of the font family. If an empty string is provided, the font will act as a fallback.
* @param bold Use `true` for bold, `false` otherwise.
* @param italic Use `true` for italic, `false` otherwise.
* @param filename The path to the font file.
* @return `true` if the font face was successfully added to the cache, `false` otherwise.
*/
LUNASVG_API bool lunasvg_add_font_face_from_file(const char* family, bool bold, bool italic, const char* filename);

/**
* @brief Add a font face from memory to the cache.
* @param family The name of the font family. If an empty string is provided, the font will act as a fallback.
* @param bold Use `true` for bold, `false` otherwise.
* @param italic Use `true` for italic, `false` otherwise.
* @param data A pointer to the memory buffer containing the font data.
* @param length The size of the memory buffer in bytes.
* @param destroy_func Callback function to free the memory buffer when it is no longer needed.
* @param closure User-defined pointer passed to the `destroy_func` callback.
* @return `true` if the font face was successfully added to the cache, `false` otherwise.
*/
LUNASVG_API bool lunasvg_add_font_face_from_data(const char* family, bool bold, bool italic, const void* data, size_t length, lunasvg_destroy_func_t destroy_func, void* closure);

#ifdef __cplusplus
}
#endif

namespace lunasvg {

/**
* @note Bitmap pixel format is ARGB32_Premultiplied.
*/
class LUNASVG_API Bitmap {
public:
    /**
     * @brief Constructs a null bitmap.
     */
    Bitmap() = default;

    /**
     * @brief Constructs a bitmap with the specified width and height.
     * @note A null bitmap will be returned if memory cannot be allocated.
     * @param width The width of the bitmap in pixels.
     * @param height The height of the bitmap in pixels.
     */
    Bitmap(int width, int height);

    /**
     * @brief Constructs a bitmap with the provided pixel data, width, height, and stride.
     *
     * @param data A pointer to the raw pixel data in ARGB32 Premultiplied format.
     * @param width The width of the bitmap in pixels.
     * @param height The height of the bitmap in pixels.
     * @param stride The number of bytes per row of pixel data (stride).
    */
    Bitmap(uint8_t* data, int width, int height, int stride);

    /**
     * @brief Copy constructor.
     * @param bitmap The bitmap to copy.
     */
    Bitmap(const Bitmap& bitmap);

    /**
     * @brief Move constructor.
     * @param bitmap The bitmap to move.
     */
    Bitmap(Bitmap&& bitmap);

    /**
     * @internal
     */
    Bitmap(plutovg_surface_t* surface) : m_surface(surface) {}

    /**
     * @brief Cleans up any resources associated with the bitmap.
     */
    ~Bitmap();

    /**
     * @brief Copy assignment operator.
     * @param bitmap The bitmap to copy.
     * @return A reference to this bitmap.
     */
    Bitmap& operator=(const Bitmap& bitmap);

    /**
     * @brief Move assignment operator.
     * @param bitmap The bitmap to move.
     * @return A reference to this bitmap.
     */
    Bitmap& operator=(Bitmap&& bitmap);

    /**
     * @brief Swaps the content of this bitmap with another.
     * @param bitmap The bitmap to swap with.
     */
    void swap(Bitmap& bitmap);

    /**
     * @brief Gets the pointer to the raw pixel data.
     * @return A pointer to the raw pixel data.
     */
    uint8_t* data() const;

    /**
     * @brief Gets the width of the bitmap.
     * @return The width of the bitmap in pixels.
     */
    int width() const;

    /**
     * @brief Gets the height of the bitmap.
     * @return The height of the bitmap in pixels.
     */
    int height() const;

    /**
     * @brief Gets the stride of the bitmap.
     * @return The number of bytes per row of pixel data (stride).
     */
    int stride() const;

    /**
     * @brief Clears the bitmap with the specified color.
     * @param The color value in 0xRRGGBBAA format.
     */
    void clear(uint32_t value);

    /**
     * @brief Converts the bitmap pixel data from ARGB32 Premultiplied to RGBA Plain format in place.
     */
    void convertToRGBA();

    /**
     * @brief Checks if the bitmap is null.
     * @return True if the bitmap is null, false otherwise.
     */
    bool isNull() const { return m_surface == nullptr; }

    /**
     * @brief Checks if the bitmap is valid.
     * @deprecated This function has been deprecated. Use `isNull()` instead to check whether the bitmap is null.
     * @return True if the bitmap is valid, false otherwise.
     */
    bool valid() const { return !isNull(); }

    /**
     * @brief Writes the bitmap to a PNG file.
     * @param filename The name of the file to write.
     * @return True if the file was written successfully, false otherwise.
     */
    bool writeToPng(const std::string& filename) const;

    /**
     * @brief Writes the bitmap to a PNG stream.
     * @param callback Callback function for writing data.
     * @param closure User-defined data passed to the callback.
     * @return True if successful, false otherwise.
     */
    bool writeToPng(lunasvg_write_func_t callback, void* closure) const;

    /**
     * @internal
     */
    plutovg_surface_t* surface() const { return m_surface; }

private:
    plutovg_surface_t* release();
    plutovg_surface_t* m_surface{nullptr};
};

class Rect;
class Matrix;

/**
 * @brief Represents a 2D axis-aligned bounding box.
 */
class LUNASVG_API Box {
public:
    /**
     * @brief Constructs a box with zero dimensions.
     */
    Box() = default;

    /**
     * @brief Constructs a box with the specified position and size.
     * @param x The x-coordinate of the box's origin.
     * @param y The y-coordinate of the box's origin.
     * @param w The width of the box.
     * @param h The height of the box.
     */
    Box(float x, float y, float w, float h);

    /**
     * @internal
     */
    Box(const Rect& rect);

    /**
     * @brief Transforms the box using the specified matrix.
     * @param matrix The transformation matrix.
     * @return A reference to this box, modified by the transformation.
     */
    Box& transform(const Matrix& matrix);

    /**
     * @brief Returns a new box transformed by the specified matrix.
     * @param matrix The transformation matrix.
     * @return A new box, transformed by the matrix.
     */
    Box transformed(const Matrix& matrix) const;

    float x{0}; ///< The x-coordinate of the box's origin.
    float y{0}; ///< The y-coordinate of the box's origin.
    float w{0}; ///< The width of the box.
    float h{0}; ///< The height of the box.
};

class Transform;

/**
 * @brief Represents a 2D transformation matrix.
 */
class LUNASVG_API Matrix {
public:
    /**
     * @brief Initializes the matrix to the identity matrix.
     */
    Matrix() = default;

    /**
     * @brief Constructs a matrix with the specified values.
     * @param a The horizontal scaling factor.
     * @param b The vertical shearing factor.
     * @param c The horizontal shearing factor.
     * @param d The vertical scaling factor.
     * @param e The horizontal translation offset.
     * @param f The vertical translation offset.
     */
    Matrix(float a, float b, float c, float d, float e, float f);

    /**
     * @internal
     */
    Matrix(const plutovg_matrix_t& matrix);

    /**
     * @internal
     */
    Matrix(const Transform& transform);

    /**
     * @brief Multiplies this matrix with another matrix.
     * @param matrix The matrix to multiply with.
     * @return A new matrix that is the result of the multiplication.
     */
    Matrix operator*(const Matrix& matrix) const;

    /**
     * @brief Multiplies this matrix with another matrix in place.
     * @param matrix The matrix to multiply with.
     * @return A reference to this matrix after multiplication.
     */
    Matrix& operator*=(const Matrix& matrix);

    /**
     * @brief Multiplies this matrix with another matrix.
     * @param matrix The matrix to multiply with.
     * @return A reference to this matrix after multiplication.
     */
    Matrix& multiply(const Matrix& matrix);

    /**
     * @brief Translates this matrix by the specified offsets.
     * @param tx The horizontal translation offset.
     * @param ty The vertical translation offset.
     * @return A reference to this matrix after translation.
     */
    Matrix& translate(float tx, float ty);

    /**
     * @brief Scales this matrix by the specified factors.
     * @param sx The horizontal scaling factor.
     * @param sy The vertical scaling factor.
     * @return A reference to this matrix after scaling.
     */
    Matrix& scale(float sx, float sy);

    /**
     * @brief Rotates this matrix by the specified angle around a point.
     * @param angle The rotation angle in degrees.
     * @param cx The x-coordinate of the center of rotation.
     * @param cy The y-coordinate of the center of rotation.
     * @return A reference to this matrix after rotation.
     */
    Matrix& rotate(float angle, float cx = 0.f, float cy = 0.f);

    /**
     * @brief Shears this matrix by the specified factors.
     * @param shx The horizontal shearing factor.
     * @param shy The vertical shearing factor.
     * @return A reference to this matrix after shearing.
     */
    Matrix& shear(float shx, float shy);

    /**
     * @brief Inverts this matrix.
     * @return A reference to this matrix after inversion.
     */
    Matrix& invert();

    /**
     * @brief Returns the inverse of this matrix.
     * @return A new matrix that is the inverse of this matrix.
     */
    Matrix inverse() const;

    /**
     * @brief Resets this matrix to the identity matrix.
     */
    void reset();

    /**
     * @brief Creates a translation matrix with the specified offsets.
     * @param tx The horizontal translation offset.
     * @param ty The vertical translation offset.
     * @return A new translation matrix.
     */
    static Matrix translated(float tx, float ty);

    /**
     * @brief Creates a scaling matrix with the specified factors.
     * @param sx The horizontal scaling factor.
     * @param sy The vertical scaling factor.
     * @return A new scaling matrix.
     */
    static Matrix scaled(float sx, float sy);

    /**
     * @brief Creates a rotation matrix with the specified angle around a point.
     * @param angle The rotation angle in degrees.
     * @param cx The x-coordinate of the center of rotation.
     * @param cy The y-coordinate of the center of rotation.
     * @return A new rotation matrix.
     */
    static Matrix rotated(float angle, float cx = 0.f, float cy = 0.f);

    /**
     * @brief Creates a shearing matrix with the specified factors.
     * @param shx The horizontal shearing factor.
     * @param shy The vertical shearing factor.
     * @return A new shearing matrix.
     */
    static Matrix sheared(float shx, float shy);

    float a{1}; ///< The horizontal scaling factor.
    float b{0}; ///< The vertical shearing factor.
    float c{0}; ///< The horizontal shearing factor.
    float d{1}; ///< The vertical scaling factor.
    float e{0}; ///< The horizontal translation offset.
    float f{0}; ///< The vertical translation offset.
};

class SVGNode;
class SVGTextNode;
class SVGElement;

class Element;
class TextNode;

class LUNASVG_API Node {
public:
    /**
     * @brief Constructs a null node.
     */
    Node() = default;

    /**
     * @brief Checks if the node is a text node.
     * @return True if the node is a text node, false otherwise.
     */
    bool isTextNode() const;

    /**
     * @brief Checks if the node is an element node.
     * @return True if the node is an element node, false otherwise.
     */
    bool isElement() const;

    /**
     * @brief Converts the node to a TextNode.
     * @return A TextNode or a null node if conversion is not possible.
     */
    TextNode toTextNode() const;

    /**
     * @brief Converts the node to an Element.
     * @return An Element or a null node if conversion is not possible.
     */
    Element toElement() const;

    /**
     * @brief Returns the parent element.
     * @return The parent element of this node. If this node has no parent, a null `Element` is returned.
     */
    Element parentElement() const;

    /**
     * @brief Checks if the node is null.
     * @return True if the node is null, false otherwise.
     */
    bool isNull() const { return m_node == nullptr; }

    /**
     * @brief Checks if two nodes are equal.
     * @param element The node to compare.
     * @return True if equal, otherwise false.
     */
    bool operator==(const Node& node) const { return m_node == node.m_node; }

    /**
     * @brief Checks if two nodes are not equal.
     * @param element The node to compare.
     * @return True if not equal, otherwise false.
     */
    bool operator!=(const Node& node) const { return m_node != node.m_node; }

protected:
    Node(SVGNode* node);
    SVGNode* node() const { return m_node; }
    SVGNode* m_node{nullptr};
    friend class Element;
};

using NodeList = std::vector<Node>;

class LUNASVG_API TextNode : public Node {
public:
    /**
     * @brief Constructs a null text node.
     */
    TextNode() = default;

    /**
     * @brief Returns the text content of the node.
     * @return A string representing the text content.
     */
    const std::string& data() const;

    /**
     * @brief Sets the text content of the node.
     * @param data The new text content to set.
     */
    void setData(const std::string& data);

private:
    TextNode(SVGTextNode* text);
    SVGTextNode* text() const;
    friend class Node;
};

class LUNASVG_API Element : public Node {
public:
    /**
     * @brief Constructs a null element.
     */
    Element() = default;

    /**
     * @brief Checks if the element has a specific attribute.
     * @param name The name of the attribute to check.
     * @return True if the element has the specified attribute, false otherwise.
     */
    bool hasAttribute(const std::string& name) const;

    /**
     * @brief Retrieves the value of an attribute.
     * @param name The name of the attribute to retrieve.
     * @return The value of the attribute as a string.
     */
    const std::string& getAttribute(const std::string& name) const;

    /**
     * @brief Sets the value of an attribute.
     * @param name The name of the attribute to set.
     * @param value The value to assign to the attribute.
     */
    void setAttribute(const std::string& name, const std::string& value);

    /**
     * @brief Renders the element onto a bitmap using a transformation matrix.
     * @param bitmap The bitmap to render onto.
     * @param The root transformation matrix.
     */
    void render(Bitmap& bitmap, const Matrix& matrix = Matrix()) const;

    /**
     * @brief Renders the element to a bitmap with specified dimensions.
     * @param width The desired width in pixels, or -1 to auto-scale based on the intrinsic size.
     * @param height The desired height in pixels, or -1 to auto-scale based on the intrinsic size.
     * @param backgroundColor The background color in 0xRRGGBBAA format.
     * @return A Bitmap containing the raster representation of the element.
     */
    Bitmap renderToBitmap(int width = -1, int height = -1, uint32_t backgroundColor = 0x00000000) const;

    /**
     * @brief Retrieves the local transformation matrix of the element.
     * @return The matrix that applies only to the element, relative to its parent.
     */
    Matrix getLocalMatrix() const;

    /**
     * @brief Retrieves the global transformation matrix of the element.
     * @return The matrix combining the element's local and all parent transformations.
     */
    Matrix getGlobalMatrix() const;

    /**
     * @brief Retrieves the local bounding box of the element.
     * @return A Box representing the bounding box after applying local transformations.
     */
    Box getLocalBoundingBox() const;

    /**
     * @brief Retrieves the global bounding box of the element.
     * @return A Box representing the bounding box after applying global transformations.
     */
    Box getGlobalBoundingBox() const;

    /**
     * @brief Retrieves the bounding box of the element without any transformations.
     * @return A Box representing the bounding box of the element without any transformations applied.
     */
    Box getBoundingBox() const;

    /**
     * @brief Returns the child nodes of this node.
     * @return A NodeList containing the child nodes.
     */
    NodeList children() const;

private:
    Element(SVGElement* element);
    SVGElement* element(bool layout = false) const;
    friend class Node;
    friend class Document;
};

class SVGRootElement;

class LUNASVG_API Document {
public:
    /**
     * @brief Load an SVG document from a file.
     * @param filename The path to the SVG file.
     * @return A pointer to the loaded `Document`, or `nullptr` on failure.
     */
    static std::unique_ptr<Document> loadFromFile(const std::string& filename);

    /**
     * @brief Load an SVG document from a string.
     * @param string The SVG data as a string.
     * @return A pointer to the loaded `Document`, or `nullptr` on failure.
     */
    static std::unique_ptr<Document> loadFromData(const std::string& string);

    /**
     * @brief Load an SVG document from a null-terminated string.
     * @param data The string containing the SVG data.
     * @return A pointer to the loaded `Document`, or `nullptr` on failure.
     */
    static std::unique_ptr<Document> loadFromData(const char* data);

    /**
     * @brief Load an SVG document from a string with a specified length.
     * @param data The string containing the SVG data.
     * @param length The length of the string in bytes.
     * @return A pointer to the loaded `Document`, or `nullptr` on failure.
     */
    static std::unique_ptr<Document> loadFromData(const char* data, size_t length);

    /**
     * @brief Applies a CSS stylesheet to the document.
     * @param content A string containing the CSS rules to apply, with comments removed.
     */
    void applyStyleSheet(const std::string& content);

    /**
     * @brief Returns the intrinsic width of the document in pixels.
     * @return The width of the document.
     */
    float width() const;

    /**
     * @brief Returns the intrinsic height of the document in pixels.
     * @return The height of the document.
     */
    float height() const;

    /**
     * @brief Returns the smallest rectangle that encloses the document content.
     * @return A Box representing the bounding box of the document.
     */
    Box boundingBox() const;

    /**
     * @brief Updates the layout of the document if needed.
     */
    void updateLayout();

    /**
     * @brief Forces an immediate layout update.
     */
    void forceLayout();

    /**
     * @brief Renders the document onto a bitmap using a transformation matrix.
     * @param bitmap The bitmap to render onto.
     * @param The root transformation matrix.
     */
    void render(Bitmap& bitmap, const Matrix& matrix = Matrix()) const;

    /**
     * @brief Renders the document to a bitmap with specified dimensions.
     * @param width The desired width in pixels, or -1 to auto-scale based on the intrinsic size.
     * @param height The desired height in pixels, or -1 to auto-scale based on the intrinsic size.
     * @param backgroundColor The background color in 0xRRGGBBAA format.
     * @return A Bitmap containing the raster representation of the document.
     */
    Bitmap renderToBitmap(int width = -1, int height = -1, uint32_t backgroundColor = 0x00000000) const;

    /**
     * @brief Retrieves an element by its ID.
     * @param id The ID of the element to retrieve.
     * @return The Element with the specified ID, or a null `Element` if not found.
     */
    Element getElementById(const std::string& id) const;

    /**
     * @brief Retrieves the document element.
     * @return The root Element of the document.
     */
    Element documentElement() const;

    /**
     * @internal
     */
    SVGRootElement* rootElement() const { return m_rootElement.get(); }

    Document(Document&&);
    Document& operator=(Document&&);
    ~Document();

private:
    Document();
    Document(const Document&) = delete;
    Document& operator=(const Document&) = delete;
    bool parse(const char* data, size_t length);
    std::unique_ptr<SVGRootElement> m_rootElement;
};

} //namespace lunasvg

#endif // LUNASVG_H
