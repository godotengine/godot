## Earcut.hpp changelog

### master

 - Fixed a bunch of rare edge cases that led to bad triangulation (parity with Earcut v2.2.2)
 - Removed use of deprecated `std::allocator::construct`
 - Fixed a minor z-order hashing bug
 - Improved visualization app, better docs

### v0.12.4

 - Fixed a crash in Crash in Earcut::findHoleBridge
 - Added coverage checks
 - Added macOS, MinGW builds

### v0.12.3

 - Fixed -Wunused-lambda-capture

### v0.12.2

 - Fixed potential division by zero
 - Fixed -fsanitize=integer warning

### v0.12.1

 - Fixed cast precision warning
