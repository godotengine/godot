#ifndef INVDYNEIGENINTERFACE_HPP_
#define INVDYNEIGENINTERFACE_HPP_
#include "../IDConfig.hpp"
namespace btInverseDynamics {

#define BT_ID_HAVE_MAT3X

#ifdef BT_USE_DOUBLE_PRECISION
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::DontAlign> vecx;
typedef Eigen::Matrix<double, 3, 1, Eigen::DontAlign> vec3;
typedef Eigen::Matrix<double, 3, 3, Eigen::DontAlign> mat33;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign> matxx;
typedef Eigen::Matrix<double, 3, Eigen::Dynamic, Eigen::DontAlign> mat3x;
#else
typedef Eigen::Matrix<float, Eigen::Dynamic, 1, Eigen::DontAlign> vecx;
typedef Eigen::Matrix<float, 3, 1, Eigen::DontAlign> vec3;
typedef Eigen::Matrix<float, 3, 3, Eigen::DontAlign> mat33;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::DontAlign> matxx;
typedef Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::DontAlign> mat3x;
#endif

inline void resize(mat3x &m, Eigen::Index size) {
    m.resize(3, size);
    m.setZero();
}

inline void setMatxxElem(const idArrayIdx row, const idArrayIdx col, const idScalar val, matxx*m){
    (*m)(row, col) = val;
}

inline void setMat3xElem(const idArrayIdx row, const idArrayIdx col, const idScalar val, mat3x*m){
    (*m)(row, col) = val;
}

}
#endif  // INVDYNEIGENINTERFACE_HPP_
