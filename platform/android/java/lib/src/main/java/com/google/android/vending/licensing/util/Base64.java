// Portions copyright 2002, Google, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.android.vending.licensing.util;

// This code was converted from code at http://iharder.sourceforge.net/base64/
// Lots of extraneous features were removed.
/* The original code said:
 * <p>
 * I am placing this code in the Public Domain. Do with it as you will.
 * This software comes with no guarantees or warranties but with
 * plenty of well-wishing instead!
 * Please visit
 * <a href="http://iharder.net/xmlizable">http://iharder.net/xmlizable</a>
 * periodically to check for updates or to contribute improvements.
 * </p>
 *
 * @author Robert Harder
 * @author rharder@usa.net
 * @version 1.3
 */

// -- GODOT start --
import org.godotengine.godot.BuildConfig;
// -- GODOT end --

/**
 * Base64 converter class. This code is not a full-blown MIME encoder;
 * it simply converts binary data to base64 data and back.
 *
 * <p>Note {@link CharBase64} is a GWT-compatible implementation of this
 * class.
 */
public class Base64 {
  /** Specify encoding (value is {@code true}). */
  public final static boolean ENCODE = true;

  /** Specify decoding (value is {@code false}). */
  public final static boolean DECODE = false;

  /** The equals sign (=) as a byte. */
  private final static byte EQUALS_SIGN = (byte) '=';

  /** The new line character (\n) as a byte. */
  private final static byte NEW_LINE = (byte) '\n';

  /**
   * The 64 valid Base64 values.
   */
  private final static byte[] ALPHABET =
      {(byte) 'A', (byte) 'B', (byte) 'C', (byte) 'D', (byte) 'E', (byte) 'F',
          (byte) 'G', (byte) 'H', (byte) 'I', (byte) 'J', (byte) 'K',
          (byte) 'L', (byte) 'M', (byte) 'N', (byte) 'O', (byte) 'P',
          (byte) 'Q', (byte) 'R', (byte) 'S', (byte) 'T', (byte) 'U',
          (byte) 'V', (byte) 'W', (byte) 'X', (byte) 'Y', (byte) 'Z',
          (byte) 'a', (byte) 'b', (byte) 'c', (byte) 'd', (byte) 'e',
          (byte) 'f', (byte) 'g', (byte) 'h', (byte) 'i', (byte) 'j',
          (byte) 'k', (byte) 'l', (byte) 'm', (byte) 'n', (byte) 'o',
          (byte) 'p', (byte) 'q', (byte) 'r', (byte) 's', (byte) 't',
          (byte) 'u', (byte) 'v', (byte) 'w', (byte) 'x', (byte) 'y',
          (byte) 'z', (byte) '0', (byte) '1', (byte) '2', (byte) '3',
          (byte) '4', (byte) '5', (byte) '6', (byte) '7', (byte) '8',
          (byte) '9', (byte) '+', (byte) '/'};

  /**
   * The 64 valid web safe Base64 values.
   */
  private final static byte[] WEBSAFE_ALPHABET =
      {(byte) 'A', (byte) 'B', (byte) 'C', (byte) 'D', (byte) 'E', (byte) 'F',
          (byte) 'G', (byte) 'H', (byte) 'I', (byte) 'J', (byte) 'K',
          (byte) 'L', (byte) 'M', (byte) 'N', (byte) 'O', (byte) 'P',
          (byte) 'Q', (byte) 'R', (byte) 'S', (byte) 'T', (byte) 'U',
          (byte) 'V', (byte) 'W', (byte) 'X', (byte) 'Y', (byte) 'Z',
          (byte) 'a', (byte) 'b', (byte) 'c', (byte) 'd', (byte) 'e',
          (byte) 'f', (byte) 'g', (byte) 'h', (byte) 'i', (byte) 'j',
          (byte) 'k', (byte) 'l', (byte) 'm', (byte) 'n', (byte) 'o',
          (byte) 'p', (byte) 'q', (byte) 'r', (byte) 's', (byte) 't',
          (byte) 'u', (byte) 'v', (byte) 'w', (byte) 'x', (byte) 'y',
          (byte) 'z', (byte) '0', (byte) '1', (byte) '2', (byte) '3',
          (byte) '4', (byte) '5', (byte) '6', (byte) '7', (byte) '8',
          (byte) '9', (byte) '-', (byte) '_'};

  /**
   * Translates a Base64 value to either its 6-bit reconstruction value
   * or a negative number indicating some other meaning.
   **/
  private final static byte[] DECODABET = {-9, -9, -9, -9, -9, -9, -9, -9, -9, // Decimal  0 -  8
      -5, -5, // Whitespace: Tab and Linefeed
      -9, -9, // Decimal 11 - 12
      -5, // Whitespace: Carriage Return
      -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, // Decimal 14 - 26
      -9, -9, -9, -9, -9, // Decimal 27 - 31
      -5, // Whitespace: Space
      -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, // Decimal 33 - 42
      62, // Plus sign at decimal 43
      -9, -9, -9, // Decimal 44 - 46
      63, // Slash at decimal 47
      52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // Numbers zero through nine
      -9, -9, -9, // Decimal 58 - 60
      -1, // Equals sign at decimal 61
      -9, -9, -9, // Decimal 62 - 64
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, // Letters 'A' through 'N'
      14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // Letters 'O' through 'Z'
      -9, -9, -9, -9, -9, -9, // Decimal 91 - 96
      26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, // Letters 'a' through 'm'
      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // Letters 'n' through 'z'
      -9, -9, -9, -9, -9 // Decimal 123 - 127
      /*  ,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 128 - 139
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 140 - 152
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 153 - 165
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 166 - 178
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 179 - 191
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 192 - 204
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 205 - 217
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 218 - 230
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 231 - 243
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9         // Decimal 244 - 255 */
      };

  /** The web safe decodabet */
  private final static byte[] WEBSAFE_DECODABET =
      {-9, -9, -9, -9, -9, -9, -9, -9, -9, // Decimal  0 -  8
          -5, -5, // Whitespace: Tab and Linefeed
          -9, -9, // Decimal 11 - 12
          -5, // Whitespace: Carriage Return
          -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, // Decimal 14 - 26
          -9, -9, -9, -9, -9, // Decimal 27 - 31
          -5, // Whitespace: Space
          -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, -9, // Decimal 33 - 44
          62, // Dash '-' sign at decimal 45
          -9, -9, // Decimal 46-47
          52, 53, 54, 55, 56, 57, 58, 59, 60, 61, // Numbers zero through nine
          -9, -9, -9, // Decimal 58 - 60
          -1, // Equals sign at decimal 61
          -9, -9, -9, // Decimal 62 - 64
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, // Letters 'A' through 'N'
          14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, // Letters 'O' through 'Z'
          -9, -9, -9, -9, // Decimal 91-94
          63, // Underscore '_' at decimal 95
          -9, // Decimal 96
          26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, // Letters 'a' through 'm'
          39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, // Letters 'n' through 'z'
          -9, -9, -9, -9, -9 // Decimal 123 - 127
      /*  ,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 128 - 139
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 140 - 152
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 153 - 165
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 166 - 178
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 179 - 191
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 192 - 204
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 205 - 217
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 218 - 230
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,     // Decimal 231 - 243
        -9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9,-9         // Decimal 244 - 255 */
      };

  // Indicates white space in encoding
  private final static byte WHITE_SPACE_ENC = -5;
  // Indicates equals sign in encoding
  private final static byte EQUALS_SIGN_ENC = -1;

  /** Defeats instantiation. */
  private Base64() {
  }

  /* ********  E N C O D I N G   M E T H O D S  ******** */

  /**
   * Encodes up to three bytes of the array <var>source</var>
   * and writes the resulting four Base64 bytes to <var>destination</var>.
   * The source and destination arrays can be manipulated
   * anywhere along their length by specifying
   * <var>srcOffset</var> and <var>destOffset</var>.
   * This method does not check to make sure your arrays
   * are large enough to accommodate <var>srcOffset</var> + 3 for
   * the <var>source</var> array or <var>destOffset</var> + 4 for
   * the <var>destination</var> array.
   * The actual number of significant bytes in your array is
   * given by <var>numSigBytes</var>.
   *
   * @param source the array to convert
   * @param srcOffset the index where conversion begins
   * @param numSigBytes the number of significant bytes in your array
   * @param destination the array to hold the conversion
   * @param destOffset the index where output will be put
   * @param alphabet is the encoding alphabet
   * @return the <var>destination</var> array
   * @since 1.3
   */
  private static byte[] encode3to4(byte[] source, int srcOffset,
      int numSigBytes, byte[] destination, int destOffset, byte[] alphabet) {
    //           1         2         3
    // 01234567890123456789012345678901 Bit position
    // --------000000001111111122222222 Array position from threeBytes
    // --------|    ||    ||    ||    | Six bit groups to index alphabet
    //          >>18  >>12  >> 6  >> 0  Right shift necessary
    //                0x3f  0x3f  0x3f  Additional AND

    // Create buffer with zero-padding if there are only one or two
    // significant bytes passed in the array.
    // We have to shift left 24 in order to flush out the 1's that appear
    // when Java treats a value as negative that is cast from a byte to an int.
    int inBuff =
        (numSigBytes > 0 ? ((source[srcOffset] << 24) >>> 8) : 0)
            | (numSigBytes > 1 ? ((source[srcOffset + 1] << 24) >>> 16) : 0)
            | (numSigBytes > 2 ? ((source[srcOffset + 2] << 24) >>> 24) : 0);

    switch (numSigBytes) {
      case 3:
        destination[destOffset] = alphabet[(inBuff >>> 18)];
        destination[destOffset + 1] = alphabet[(inBuff >>> 12) & 0x3f];
        destination[destOffset + 2] = alphabet[(inBuff >>> 6) & 0x3f];
        destination[destOffset + 3] = alphabet[(inBuff) & 0x3f];
        return destination;
      case 2:
        destination[destOffset] = alphabet[(inBuff >>> 18)];
        destination[destOffset + 1] = alphabet[(inBuff >>> 12) & 0x3f];
        destination[destOffset + 2] = alphabet[(inBuff >>> 6) & 0x3f];
        destination[destOffset + 3] = EQUALS_SIGN;
        return destination;
      case 1:
        destination[destOffset] = alphabet[(inBuff >>> 18)];
        destination[destOffset + 1] = alphabet[(inBuff >>> 12) & 0x3f];
        destination[destOffset + 2] = EQUALS_SIGN;
        destination[destOffset + 3] = EQUALS_SIGN;
        return destination;
      default:
        return destination;
    } // end switch
  } // end encode3to4

  /**
   * Encodes a byte array into Base64 notation.
   * Equivalent to calling
   * {@code encodeBytes(source, 0, source.length)}
   *
   * @param source The data to convert
   * @since 1.4
   */
  public static String encode(byte[] source) {
    return encode(source, 0, source.length, ALPHABET, true);
  }

  /**
   * Encodes a byte array into web safe Base64 notation.
   *
   * @param source The data to convert
   * @param doPadding is {@code true} to pad result with '=' chars
   *        if it does not fall on 3 byte boundaries
   */
  public static String encodeWebSafe(byte[] source, boolean doPadding) {
    return encode(source, 0, source.length, WEBSAFE_ALPHABET, doPadding);
  }

  /**
   * Encodes a byte array into Base64 notation.
   *
   * @param source The data to convert
   * @param off Offset in array where conversion should begin
   * @param len Length of data to convert
   * @param alphabet is the encoding alphabet
   * @param doPadding is {@code true} to pad result with '=' chars
   *        if it does not fall on 3 byte boundaries
   * @since 1.4
   */
  public static String encode(byte[] source, int off, int len, byte[] alphabet,
      boolean doPadding) {
    byte[] outBuff = encode(source, off, len, alphabet, Integer.MAX_VALUE);
    int outLen = outBuff.length;

    // If doPadding is false, set length to truncate '='
    // padding characters
    while (doPadding == false && outLen > 0) {
      if (outBuff[outLen - 1] != '=') {
        break;
      }
      outLen -= 1;
    }

    return new String(outBuff, 0, outLen);
  }

  /**
   * Encodes a byte array into Base64 notation.
   *
   * @param source The data to convert
   * @param off Offset in array where conversion should begin
   * @param len Length of data to convert
   * @param alphabet is the encoding alphabet
   * @param maxLineLength maximum length of one line.
   * @return the BASE64-encoded byte array
   */
  public static byte[] encode(byte[] source, int off, int len, byte[] alphabet,
      int maxLineLength) {
    int lenDiv3 = (len + 2) / 3; // ceil(len / 3)
    int len43 = lenDiv3 * 4;
    byte[] outBuff = new byte[len43 // Main 4:3
        + (len43 / maxLineLength)]; // New lines

    int d = 0;
    int e = 0;
    int len2 = len - 2;
    int lineLength = 0;
    for (; d < len2; d += 3, e += 4) {

      // The following block of code is the same as
      // encode3to4( source, d + off, 3, outBuff, e, alphabet );
      // but inlined for faster encoding (~20% improvement)
      int inBuff =
          ((source[d + off] << 24) >>> 8)
              | ((source[d + 1 + off] << 24) >>> 16)
              | ((source[d + 2 + off] << 24) >>> 24);
      outBuff[e] = alphabet[(inBuff >>> 18)];
      outBuff[e + 1] = alphabet[(inBuff >>> 12) & 0x3f];
      outBuff[e + 2] = alphabet[(inBuff >>> 6) & 0x3f];
      outBuff[e + 3] = alphabet[(inBuff) & 0x3f];

      lineLength += 4;
      if (lineLength == maxLineLength) {
        outBuff[e + 4] = NEW_LINE;
        e++;
        lineLength = 0;
      } // end if: end of line
    } // end for: each piece of array

    if (d < len) {
      encode3to4(source, d + off, len - d, outBuff, e, alphabet);

      lineLength += 4;
      if (lineLength == maxLineLength) {
        // Add a last newline
        outBuff[e + 4] = NEW_LINE;
        e++;
      }
      e += 4;
    }

    // -- GODOT start --
    //assert (e == outBuff.length);
    if (BuildConfig.DEBUG && e != outBuff.length)
      throw new RuntimeException();
    // -- GODOT end --
    return outBuff;
  }


  /* ********  D E C O D I N G   M E T H O D S  ******** */


  /**
   * Decodes four bytes from array <var>source</var>
   * and writes the resulting bytes (up to three of them)
   * to <var>destination</var>.
   * The source and destination arrays can be manipulated
   * anywhere along their length by specifying
   * <var>srcOffset</var> and <var>destOffset</var>.
   * This method does not check to make sure your arrays
   * are large enough to accommodate <var>srcOffset</var> + 4 for
   * the <var>source</var> array or <var>destOffset</var> + 3 for
   * the <var>destination</var> array.
   * This method returns the actual number of bytes that
   * were converted from the Base64 encoding.
   *
   *
   * @param source the array to convert
   * @param srcOffset the index where conversion begins
   * @param destination the array to hold the conversion
   * @param destOffset the index where output will be put
   * @param decodabet the decodabet for decoding Base64 content
   * @return the number of decoded bytes converted
   * @since 1.3
   */
  private static int decode4to3(byte[] source, int srcOffset,
      byte[] destination, int destOffset, byte[] decodabet) {
    // Example: Dk==
    if (source[srcOffset + 2] == EQUALS_SIGN) {
      int outBuff =
          ((decodabet[source[srcOffset]] << 24) >>> 6)
              | ((decodabet[source[srcOffset + 1]] << 24) >>> 12);

      destination[destOffset] = (byte) (outBuff >>> 16);
      return 1;
    } else if (source[srcOffset + 3] == EQUALS_SIGN) {
      // Example: DkL=
      int outBuff =
          ((decodabet[source[srcOffset]] << 24) >>> 6)
              | ((decodabet[source[srcOffset + 1]] << 24) >>> 12)
              | ((decodabet[source[srcOffset + 2]] << 24) >>> 18);

      destination[destOffset] = (byte) (outBuff >>> 16);
      destination[destOffset + 1] = (byte) (outBuff >>> 8);
      return 2;
    } else {
      // Example: DkLE
      int outBuff =
          ((decodabet[source[srcOffset]] << 24) >>> 6)
              | ((decodabet[source[srcOffset + 1]] << 24) >>> 12)
              | ((decodabet[source[srcOffset + 2]] << 24) >>> 18)
              | ((decodabet[source[srcOffset + 3]] << 24) >>> 24);

      destination[destOffset] = (byte) (outBuff >> 16);
      destination[destOffset + 1] = (byte) (outBuff >> 8);
      destination[destOffset + 2] = (byte) (outBuff);
      return 3;
    }
  } // end decodeToBytes


  /**
   * Decodes data from Base64 notation.
   *
   * @param s the string to decode (decoded in default encoding)
   * @return the decoded data
   * @since 1.4
   */
  public static byte[] decode(String s) throws Base64DecoderException {
    byte[] bytes = s.getBytes();
    return decode(bytes, 0, bytes.length);
  }

  /**
   * Decodes data from web safe Base64 notation.
   * Web safe encoding uses '-' instead of '+', '_' instead of '/'
   *
   * @param s the string to decode (decoded in default encoding)
   * @return the decoded data
   */
  public static byte[] decodeWebSafe(String s) throws Base64DecoderException {
    byte[] bytes = s.getBytes();
    return decodeWebSafe(bytes, 0, bytes.length);
  }

  /**
   * Decodes Base64 content in byte array format and returns
   * the decoded byte array.
   *
   * @param source The Base64 encoded data
   * @return decoded data
   * @since 1.3
   * @throws Base64DecoderException
   */
  public static byte[] decode(byte[] source) throws Base64DecoderException {
    return decode(source, 0, source.length);
  }

  /**
   * Decodes web safe Base64 content in byte array format and returns
   * the decoded data.
   * Web safe encoding uses '-' instead of '+', '_' instead of '/'
   *
   * @param source the string to decode (decoded in default encoding)
   * @return the decoded data
   */
  public static byte[] decodeWebSafe(byte[] source)
      throws Base64DecoderException {
    return decodeWebSafe(source, 0, source.length);
  }

  /**
   * Decodes Base64 content in byte array format and returns
   * the decoded byte array.
   *
   * @param source The Base64 encoded data
   * @param off    The offset of where to begin decoding
   * @param len    The length of characters to decode
   * @return decoded data
   * @since 1.3
   * @throws Base64DecoderException
   */
  public static byte[] decode(byte[] source, int off, int len)
      throws Base64DecoderException {
    return decode(source, off, len, DECODABET);
  }

  /**
   * Decodes web safe Base64 content in byte array format and returns
   * the decoded byte array.
   * Web safe encoding uses '-' instead of '+', '_' instead of '/'
   *
   * @param source The Base64 encoded data
   * @param off    The offset of where to begin decoding
   * @param len    The length of characters to decode
   * @return decoded data
   */
  public static byte[] decodeWebSafe(byte[] source, int off, int len)
      throws Base64DecoderException {
    return decode(source, off, len, WEBSAFE_DECODABET);
  }

  /**
   * Decodes Base64 content using the supplied decodabet and returns
   * the decoded byte array.
   *
   * @param source    The Base64 encoded data
   * @param off       The offset of where to begin decoding
   * @param len       The length of characters to decode
   * @param decodabet the decodabet for decoding Base64 content
   * @return decoded data
   */
  public static byte[] decode(byte[] source, int off, int len, byte[] decodabet)
      throws Base64DecoderException {
    int len34 = len * 3 / 4;
    byte[] outBuff = new byte[2 + len34]; // Upper limit on size of output
    int outBuffPosn = 0;

    byte[] b4 = new byte[4];
    int b4Posn = 0;
    int i = 0;
    byte sbiCrop = 0;
    byte sbiDecode = 0;
    for (i = 0; i < len; i++) {
      sbiCrop = (byte) (source[i + off] & 0x7f); // Only the low seven bits
      sbiDecode = decodabet[sbiCrop];

      if (sbiDecode >= WHITE_SPACE_ENC) { // White space Equals sign or better
        if (sbiDecode >= EQUALS_SIGN_ENC) {
          // An equals sign (for padding) must not occur at position 0 or 1
          // and must be the last byte[s] in the encoded value
          if (sbiCrop == EQUALS_SIGN) {
            int bytesLeft = len - i;
            byte lastByte = (byte) (source[len - 1 + off] & 0x7f);
            if (b4Posn == 0 || b4Posn == 1) {
              throw new Base64DecoderException(
                  "invalid padding byte '=' at byte offset " + i);
            } else if ((b4Posn == 3 && bytesLeft > 2)
                || (b4Posn == 4 && bytesLeft > 1)) {
              throw new Base64DecoderException(
                  "padding byte '=' falsely signals end of encoded value "
                      + "at offset " + i);
            } else if (lastByte != EQUALS_SIGN && lastByte != NEW_LINE) {
              throw new Base64DecoderException(
                  "encoded value has invalid trailing byte");
            }
            break;
          }

          b4[b4Posn++] = sbiCrop;
          if (b4Posn == 4) {
            outBuffPosn += decode4to3(b4, 0, outBuff, outBuffPosn, decodabet);
            b4Posn = 0;
          }
        }
      } else {
        throw new Base64DecoderException("Bad Base64 input character at " + i
            + ": " + source[i + off] + "(decimal)");
      }
    }

    // Because web safe encoding allows non padding base64 encodes, we
    // need to pad the rest of the b4 buffer with equal signs when
    // b4Posn != 0.  There can be at most 2 equal signs at the end of
    // four characters, so the b4 buffer must have two or three
    // characters.  This also catches the case where the input is
    // padded with EQUALS_SIGN
    if (b4Posn != 0) {
      if (b4Posn == 1) {
        throw new Base64DecoderException("single trailing character at offset "
            + (len - 1));
      }
      b4[b4Posn++] = EQUALS_SIGN;
      outBuffPosn += decode4to3(b4, 0, outBuff, outBuffPosn, decodabet);
    }

    byte[] out = new byte[outBuffPosn];
    System.arraycopy(outBuff, 0, out, 0, outBuffPosn);
    return out;
  }
}
