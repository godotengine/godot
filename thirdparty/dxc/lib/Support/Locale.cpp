
#include "llvm/Support/Locale.h"
#include "llvm/Support/Unicode.h"

namespace llvm {
namespace sys {
namespace locale {

int columnWidth(StringRef Text) {
#if LLVM_ON_WIN32
  return Text.size();
#else
  return llvm::sys::unicode::columnWidthUTF8(Text);
#endif
}

bool isPrint(int UCS) {
#if LLVM_ON_WIN32
  // Restrict characters that we'll try to print to the lower part of ASCII
  // except for the control characters (0x20 - 0x7E). In general one can not
  // reliably output code points U+0080 and higher using narrow character C/C++
  // output functions in Windows, because the meaning of the upper 128 codes is
  // determined by the active code page in the console.
  return ' ' <= UCS && UCS <= '~';
#else
  return llvm::sys::unicode::isPrintable(UCS);
#endif
}

} // namespace locale
} // namespace sys
} // namespace llvm
