/**************************************************************************/
/*  TestClass.kt                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

package com.godot.game.test.javaclasswrapper

import org.godotengine.godot.Dictionary
import kotlin.collections.contentToString
import kotlin.collections.joinToString

class TestClass {
	@JvmField var boolField: Boolean = false
	@JvmField var byteField: Byte = 2
	@JvmField var charField: Char = 'B'
	@JvmField var shortField: Short = 3
	@JvmField var intField: Int = 4
	@JvmField var longField: Long = 5L
	@JvmField var floatField: Float = 6.0f
	@JvmField var doubleField: Double = 7.0
	@JvmField var stringField: String = "string field"

	@JvmField var boxedBoolField: Boolean? = null
	@JvmField var boxedByteField: Byte? = null
	@JvmField var boxedCharField: Char? = null
	@JvmField var boxedShortField: Short? = null
	@JvmField var boxedIntField: Int? = null
	@JvmField var boxedLongField: Long? = null
	@JvmField var boxedFloatField: Float? = null
	@JvmField var boxedDoubleField: Double? = null

	@JvmField val finalIntField: Int = 100
	@JvmField val finalStringField: String = "final"

	companion object {
		@JvmField var staticBoolField: Boolean = true
		@JvmField var staticByteField: Byte = 1
		@JvmField var staticCharField: Char = 'A'
		@JvmField var staticShortField: Short = 2
		@JvmField var staticIntField: Int = 3
		@JvmField var staticLongField: Long = 4L
		@JvmField var staticFloatField: Float = 5.0f
		@JvmField var staticDoubleField: Double = 6.0
		@JvmField var staticStringField: String = "static"

		@JvmField val staticFinalIntField: Int = 200
		@JvmField val staticFinalStringField: String = "static final"

		@JvmStatic
		fun stringify(value: Any?): String {
			return when (value) {
				null -> "null"
				is Map<*, *> -> {
					val entries = value.entries.joinToString(", ") { (k, v) -> "${stringify(k)}: ${stringify(v)}" }
					"{$entries}"
				}

				is List<*> -> value.joinToString(prefix = "[", postfix = "]") { stringify(it) }
				is Array<*> -> value.joinToString(prefix = "[", postfix = "]") { stringify(it) }
				is IntArray -> value.joinToString(prefix = "[", postfix = "]")
				is LongArray -> value.joinToString(prefix = "[", postfix = "]")
				is FloatArray -> value.joinToString(prefix = "[", postfix = "]")
				is DoubleArray -> value.joinToString(prefix = "[", postfix = "]")
				is BooleanArray -> value.joinToString(prefix = "[", postfix = "]")
				is CharArray -> value.joinToString(prefix = "[", postfix = "]")
				else -> value.toString()
			}
		}

		@JvmStatic
		fun testDictionary(d: Dictionary): String {
			return d.toString()
		}

		@JvmStatic
		fun testDictionaryNested(d: Dictionary): String {
			return stringify(d)
		}

		@JvmStatic
		fun testRetDictionary(): Dictionary {
			var d = Dictionary()
			d.putAll(mapOf("a" to 1, "b" to 2))
			return d
		}

		@JvmStatic
		fun testRetDictionaryArray(): Array<Dictionary> {
			var d = Dictionary()
			d.putAll(mapOf("a" to 1, "b" to 2))
			return arrayOf(d)
		}

		@JvmStatic
		fun testMethod(int: Int, array: IntArray): String {
			return "IntArray: " + array.contentToString()
		}

		@JvmStatic
		fun testMethod(int: Int, vararg args: String): String {
			return "StringArray: " + args.contentToString()
		}

		@JvmStatic
		fun testMethod(int: Int, objects: Array<TestClass2>): String {
			return "testObjects: " + objects.joinToString(separator = " ") { it.getValue().toString() }
		}

		@JvmStatic
		fun testExc(i: Int): Int {
			val s: String? = null
			s!!.length
			return i
		}

		@JvmStatic
		fun testArgLong(a: Long): String {
			return "${a}"
		}

		@JvmStatic
		fun testArgBoolArray(a: BooleanArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testArgByteArray(a: ByteArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testArgCharArray(a: CharArray): String {
			return a.joinToString("")
		}

		@JvmStatic
		fun testArgShortArray(a: ShortArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testArgIntArray(a: IntArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testArgLongArray(a: LongArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testArgFloatArray(a: FloatArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testArgDoubleArray(a: DoubleArray): String {
			return a.contentToString();
		}

		@JvmStatic
		fun testRetBoolArray(): BooleanArray {
			return booleanArrayOf(true, false, true)
		}

		@JvmStatic
		fun testRetByteArray(): ByteArray {
			return byteArrayOf(1, 2, 3)
		}

		@JvmStatic
		fun testRetCharArray(): CharArray {
			return "abc".toCharArray()
		}

		@JvmStatic
		fun testRetShortArray(): ShortArray {
			return shortArrayOf(11, 12, 13)
		}

		@JvmStatic
		fun testRetIntArray(): IntArray {
			return intArrayOf(21, 22, 23)
		}

		@JvmStatic
		fun testRetLongArray(): LongArray {
			return longArrayOf(41, 42, 43)
		}

		@JvmStatic
		fun testRetFloatArray(): FloatArray {
			return floatArrayOf(31.1f, 32.2f, 33.3f)
		}

		@JvmStatic
		fun testRetDoubleArray(): DoubleArray {
			return doubleArrayOf(41.1, 42.2, 43.3)
		}

		@JvmStatic
		fun testRetWrappedBoolArray(): Array<Boolean> {
			return arrayOf(true, false, true)
		}

		@JvmStatic
		fun testRetWrappedByteArray(): Array<Byte> {
			return arrayOf(1, 2, 3)
		}

		@JvmStatic
		fun testRetWrappedCharArray(): Array<Char> {
			return arrayOf('a', 'b', 'c')
		}

		@JvmStatic
		fun testRetWrappedShortArray(): Array<Short> {
			return arrayOf(11, 12, 13)
		}

		@JvmStatic
		fun testRetWrappedIntArray(): Array<Int> {
			return arrayOf(21, 22, 23)
		}

		@JvmStatic
		fun testRetWrappedLongArray(): Array<Long> {
			return arrayOf(41, 42, 43)
		}

		@JvmStatic
		fun testRetWrappedFloatArray(): Array<Float> {
			return arrayOf(31.1f, 32.2f, 33.3f)
		}

		@JvmStatic
		fun testRetWrappedDoubleArray(): Array<Double> {
			return arrayOf(41.1, 42.2, 43.3)
		}

		@JvmStatic
		fun testRetObjectArray(): Array<TestClass2> {
			return arrayOf(TestClass2(51), TestClass2(52));
		}

		@JvmStatic
		fun testRetStringArray(): Array<String> {
			return arrayOf("I", "am", "String")
		}

		@JvmStatic
		fun testRetCharSequenceArray(): Array<CharSequence> {
			return arrayOf("I", "am", "CharSequence")
		}

		@JvmStatic
		fun testObjectOverload(a: TestClass2): String {
			return "TestClass2: $a"
		}

		@JvmStatic
		fun testObjectOverload(a: TestClass3): String {
			return "TestClass3: $a"
		}

		@JvmStatic
		fun testObjectOverloadArray(a: Array<TestClass2>): String {
			return "TestClass2: " + a.contentToString()
		}

		@JvmStatic
		fun testObjectOverloadArray(a: Array<TestClass3>): String {
			return "TestClass3: " + a.contentToString()
		}
	}
}
