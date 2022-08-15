//===---- llvm/TypeBuilder.h - Builder for LLVM types -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TypeBuilder class, which is used as a convenient way to
// create LLVM types with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_TYPEBUILDER_H
#define LLVM_IR_TYPEBUILDER_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/LLVMContext.h"
#include <climits>

namespace llvm {

/// TypeBuilder - This provides a uniform API for looking up types
/// known at compile time.  To support cross-compilation, we define a
/// series of tag types in the llvm::types namespace, like i<N>,
/// ieee_float, ppc_fp128, etc.  TypeBuilder<T, false> allows T to be
/// any of these, a native C type (whose size may depend on the host
/// compiler), or a pointer, function, or struct type built out of
/// these.  TypeBuilder<T, true> removes native C types from this set
/// to guarantee that its result is suitable for cross-compilation.
/// We define the primitive types, pointer types, and functions up to
/// 5 arguments here, but to use this class with your own types,
/// you'll need to specialize it.  For example, say you want to call a
/// function defined externally as:
///
/// \code{.cpp}
///
///   struct MyType {
///     int32 a;
///     int32 *b;
///     void *array[1];  // Intended as a flexible array.
///   };
///   int8 AFunction(struct MyType *value);
///
/// \endcode
///
/// You'll want to use
///   Function::Create(TypeBuilder<types::i<8>(MyType*), true>::get(), ...)
/// to declare the function, but when you first try this, your compiler will
/// complain that TypeBuilder<MyType, true>::get() doesn't exist. To fix this,
/// write:
///
/// \code{.cpp}
///
///   namespace llvm {
///   template<bool xcompile> class TypeBuilder<MyType, xcompile> {
///   public:
///     static StructType *get(LLVMContext &Context) {
///       // If you cache this result, be sure to cache it separately
///       // for each LLVMContext.
///       return StructType::get(
///         TypeBuilder<types::i<32>, xcompile>::get(Context),
///         TypeBuilder<types::i<32>*, xcompile>::get(Context),
///         TypeBuilder<types::i<8>*[], xcompile>::get(Context),
///         nullptr);
///     }
///
///     // You may find this a convenient place to put some constants
///     // to help with getelementptr.  They don't have any effect on
///     // the operation of TypeBuilder.
///     enum Fields {
///       FIELD_A,
///       FIELD_B,
///       FIELD_ARRAY
///     };
///   }
///   }  // namespace llvm
///
/// \endcode
///
/// TypeBuilder cannot handle recursive types or types you only know at runtime.
/// If you try to give it a recursive type, it will deadlock, infinitely
/// recurse, or do something similarly undesirable.
template<typename T, bool cross_compilable> class TypeBuilder {};

// Types for use with cross-compilable TypeBuilders.  These correspond
// exactly with an LLVM-native type.
namespace types {
/// i<N> corresponds to the LLVM IntegerType with N bits.
template<uint32_t num_bits> class i {};

// The following classes represent the LLVM floating types.
class ieee_float {};
class ieee_double {};
class x86_fp80 {};
class fp128 {};
class ppc_fp128 {};
// X86 MMX.
class x86_mmx {};
}  // namespace types

// LLVM doesn't have const or volatile types.
template<typename T, bool cross> class TypeBuilder<const T, cross>
  : public TypeBuilder<T, cross> {};
template<typename T, bool cross> class TypeBuilder<volatile T, cross>
  : public TypeBuilder<T, cross> {};
template<typename T, bool cross> class TypeBuilder<const volatile T, cross>
  : public TypeBuilder<T, cross> {};

// Pointers
template<typename T, bool cross> class TypeBuilder<T*, cross> {
public:
  static PointerType *get(LLVMContext &Context) {
    return PointerType::getUnqual(TypeBuilder<T,cross>::get(Context));
  }
};

/// There is no support for references
template<typename T, bool cross> class TypeBuilder<T&, cross> {};

// Arrays
template<typename T, size_t N, bool cross> class TypeBuilder<T[N], cross> {
public:
  static ArrayType *get(LLVMContext &Context) {
    return ArrayType::get(TypeBuilder<T, cross>::get(Context), N);
  }
};
/// LLVM uses an array of length 0 to represent an unknown-length array.
template<typename T, bool cross> class TypeBuilder<T[], cross> {
public:
  static ArrayType *get(LLVMContext &Context) {
    return ArrayType::get(TypeBuilder<T, cross>::get(Context), 0);
  }
};

// Define the C integral types only for TypeBuilder<T, false>.
//
// C integral types do not have a defined size. It would be nice to use the
// stdint.h-defined typedefs that do have defined sizes, but we'd run into the
// following problem:
//
// On an ILP32 machine, stdint.h might define:
//
//   typedef int int32_t;
//   typedef long long int64_t;
//   typedef long size_t;
//
// If we defined TypeBuilder<int32_t> and TypeBuilder<int64_t>, then any use of
// TypeBuilder<size_t> would fail.  We couldn't define TypeBuilder<size_t> in
// addition to the defined-size types because we'd get duplicate definitions on
// platforms where stdint.h instead defines:
//
//   typedef int int32_t;
//   typedef long long int64_t;
//   typedef int size_t;
//
// So we define all the primitive C types and nothing else.
#define DEFINE_INTEGRAL_TYPEBUILDER(T) \
template<> class TypeBuilder<T, false> { \
public: \
  static IntegerType *get(LLVMContext &Context) { \
    return IntegerType::get(Context, sizeof(T) * CHAR_BIT); \
  } \
}; \
template<> class TypeBuilder<T, true> { \
  /* We provide a definition here so users don't accidentally */ \
  /* define these types to work. */ \
}
DEFINE_INTEGRAL_TYPEBUILDER(char);
DEFINE_INTEGRAL_TYPEBUILDER(signed char);
DEFINE_INTEGRAL_TYPEBUILDER(unsigned char);
DEFINE_INTEGRAL_TYPEBUILDER(short);
DEFINE_INTEGRAL_TYPEBUILDER(unsigned short);
DEFINE_INTEGRAL_TYPEBUILDER(int);
DEFINE_INTEGRAL_TYPEBUILDER(unsigned int);
DEFINE_INTEGRAL_TYPEBUILDER(long);
DEFINE_INTEGRAL_TYPEBUILDER(unsigned long);
#ifdef _MSC_VER
DEFINE_INTEGRAL_TYPEBUILDER(__int64);
DEFINE_INTEGRAL_TYPEBUILDER(unsigned __int64);
#else /* _MSC_VER */
DEFINE_INTEGRAL_TYPEBUILDER(long long);
DEFINE_INTEGRAL_TYPEBUILDER(unsigned long long);
#endif /* _MSC_VER */
#undef DEFINE_INTEGRAL_TYPEBUILDER

template<uint32_t num_bits, bool cross>
class TypeBuilder<types::i<num_bits>, cross> {
public:
  static IntegerType *get(LLVMContext &C) {
    return IntegerType::get(C, num_bits);
  }
};

template<> class TypeBuilder<float, false> {
public:
  static Type *get(LLVMContext& C) {
    return Type::getFloatTy(C);
  }
};
template<> class TypeBuilder<float, true> {};

template<> class TypeBuilder<double, false> {
public:
  static Type *get(LLVMContext& C) {
    return Type::getDoubleTy(C);
  }
};
template<> class TypeBuilder<double, true> {};

template<bool cross> class TypeBuilder<types::ieee_float, cross> {
public:
  static Type *get(LLVMContext& C) { return Type::getFloatTy(C); }
};
template<bool cross> class TypeBuilder<types::ieee_double, cross> {
public:
  static Type *get(LLVMContext& C) { return Type::getDoubleTy(C); }
};
template<bool cross> class TypeBuilder<types::x86_fp80, cross> {
public:
  static Type *get(LLVMContext& C) { return Type::getX86_FP80Ty(C); }
};
template<bool cross> class TypeBuilder<types::fp128, cross> {
public:
  static Type *get(LLVMContext& C) { return Type::getFP128Ty(C); }
};
template<bool cross> class TypeBuilder<types::ppc_fp128, cross> {
public:
  static Type *get(LLVMContext& C) { return Type::getPPC_FP128Ty(C); }
};
template<bool cross> class TypeBuilder<types::x86_mmx, cross> {
public:
  static Type *get(LLVMContext& C) { return Type::getX86_MMXTy(C); }
};

template<bool cross> class TypeBuilder<void, cross> {
public:
  static Type *get(LLVMContext &C) {
    return Type::getVoidTy(C);
  }
};

/// void* is disallowed in LLVM types, but it occurs often enough in C code that
/// we special case it.
template<> class TypeBuilder<void*, false>
  : public TypeBuilder<types::i<8>*, false> {};
template<> class TypeBuilder<const void*, false>
  : public TypeBuilder<types::i<8>*, false> {};
template<> class TypeBuilder<volatile void*, false>
  : public TypeBuilder<types::i<8>*, false> {};
template<> class TypeBuilder<const volatile void*, false>
  : public TypeBuilder<types::i<8>*, false> {};

template<typename R, bool cross> class TypeBuilder<R(), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    return FunctionType::get(TypeBuilder<R, cross>::get(Context), false);
  }
};
template<typename R, typename A1, bool cross> class TypeBuilder<R(A1), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, false);
  }
};
template<typename R, typename A1, typename A2, bool cross>
class TypeBuilder<R(A1, A2), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, false);
  }
};
template<typename R, typename A1, typename A2, typename A3, bool cross>
class TypeBuilder<R(A1, A2, A3), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, false);
  }
};

template<typename R, typename A1, typename A2, typename A3, typename A4,
         bool cross>
class TypeBuilder<R(A1, A2, A3, A4), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
      TypeBuilder<A4, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, false);
  }
};

template<typename R, typename A1, typename A2, typename A3, typename A4,
         typename A5, bool cross>
class TypeBuilder<R(A1, A2, A3, A4, A5), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
      TypeBuilder<A4, cross>::get(Context),
      TypeBuilder<A5, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, false);
  }
};

template<typename R, bool cross> class TypeBuilder<R(...), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    return FunctionType::get(TypeBuilder<R, cross>::get(Context), true);
  }
};
template<typename R, typename A1, bool cross>
class TypeBuilder<R(A1, ...), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context), params, true);
  }
};
template<typename R, typename A1, typename A2, bool cross>
class TypeBuilder<R(A1, A2, ...), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                                   params, true);
  }
};
template<typename R, typename A1, typename A2, typename A3, bool cross>
class TypeBuilder<R(A1, A2, A3, ...), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                                   params, true);
  }
};

template<typename R, typename A1, typename A2, typename A3, typename A4,
         bool cross>
class TypeBuilder<R(A1, A2, A3, A4, ...), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
      TypeBuilder<A4, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                             params, true);
  }
};

template<typename R, typename A1, typename A2, typename A3, typename A4,
         typename A5, bool cross>
class TypeBuilder<R(A1, A2, A3, A4, A5, ...), cross> {
public:
  static FunctionType *get(LLVMContext &Context) {
    Type *params[] = {
      TypeBuilder<A1, cross>::get(Context),
      TypeBuilder<A2, cross>::get(Context),
      TypeBuilder<A3, cross>::get(Context),
      TypeBuilder<A4, cross>::get(Context),
      TypeBuilder<A5, cross>::get(Context),
    };
    return FunctionType::get(TypeBuilder<R, cross>::get(Context),
                                   params, true);
  }
};

}  // namespace llvm

#endif
