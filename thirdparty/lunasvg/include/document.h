#ifndef DOCUMENT_H
#define DOCUMENT_H

#include <memory>
#include <string>

namespace lunasvg {

class Bitmap
{
public:
    /**
     * @note Default bitmap format is RGBA (non-premultiplied).
     */
    Bitmap();
    Bitmap(std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint32_t stride);
    Bitmap(std::uint32_t width, std::uint32_t height);

    void reset(std::uint8_t* data, std::uint32_t width, std::uint32_t height, std::uint32_t stride);
    void reset(std::uint32_t width, std::uint32_t height);

    std::uint8_t* data() const;
    std::uint32_t width() const;
    std::uint32_t height() const;
    std::uint32_t stride() const;
    bool valid() const;

private:
    struct Impl;
    std::shared_ptr<Impl> m_impl;
};

class Box
{
public:
    Box() = default;
    Box(double x, double y, double w, double h);

public:
    double x{0};
    double y{0};
    double w{0};
    double h{0};
};

class Matrix
{
public:
    Matrix() = default;
    Matrix(double a, double b, double c, double d, double e, double f);

public:
    double a{1};
    double b{0};
    double c{0};
    double d{1};
    double e{0};
    double f{0};
};

class LayoutRoot;

class Document
{
public:
    /**
     * @brief loadFromFile
     * @param filename
     * @return
     */
    static std::unique_ptr<Document> loadFromFile(const std::string& filename);

    /**
     * @brief loadFromData
     * @param string
     * @return
     */
    static std::unique_ptr<Document> loadFromData(const std::string& string);

    /**
     * @brief loadFromData
     * @param data
     * @param size
     * @return
     */
    static std::unique_ptr<Document> loadFromData(const char* data, std::size_t size);

    /**
     * @brief loadFromData
     * @param data
     * @return
     */
    static std::unique_ptr<Document> loadFromData(const char* data);

    /**
     * @brief rotate
     * @param angle
     * @return
     */
    Document* rotate(double angle);

    /**
     * @brief rotate
     * @param angle
     * @param cx
     * @param cy
     * @return
     */
    Document* rotate(double angle, double cx, double cy);

    /**
     * @brief scale
     * @param sx
     * @param sy
     * @return
     */
    Document* scale(double sx, double sy);

    /**
     * @brief shear
     * @param shx
     * @param shy
     * @return
     */
    Document* shear(double shx, double shy);

    /**
     * @brief translate
     * @param tx
     * @param ty
     * @return
     */
    Document* translate(double tx, double ty);

    /**
     * @brief transform
     * @param a
     * @param b
     * @param c
     * @param d
     * @param e
     * @param f
     * @return
     */
    Document* transform(double a, double b, double c, double d, double e, double f);

    /**
     * @brief identity
     * @return
     */
    Document* identity();

    /**
     * @brief matrix
     * @return
     */
    Matrix matrix() const;

    /**
     * @brief box
     * @return
     */
    Box box() const;

    /**
     * @brief width
     * @return
     */
    double width() const;

    /**
     * @brief height
     * @return
     */
    double height() const;

    /**
     * @brief render
     * @param bitmap
     * @param bgColor
     */
    void render(Bitmap bitmap, std::uint32_t bgColor = 0x00000000) const;

    /**
     * @brief renderToBitmap
     * @param width
     * @param height
     * @param bgColor
     * @return
     */
    Bitmap renderToBitmap(std::uint32_t width = 0, std::uint32_t height = 0, std::uint32_t bgColor = 0x00000000) const;

    ~Document();
private:
    Document();

    std::unique_ptr<LayoutRoot> root;
};

} //namespace lunasvg

#endif // DOCUMENT_H
