/**
 * pugixml parser - version 1.13
 * --------------------------------------------------------
 * Copyright (C) 2006-2022, by Arseny Kapoulkine (arseny.kapoulkine@gmail.com)
 * Report bugs and download new versions at https://pugixml.org/
 *
 * This library is distributed under the MIT License. See notice at the end
 * of this file.
 *
 * This work is based on the pugxml parser, which is:
 * Copyright (C) 2003, by Kristen Wegner (kristen@tima.net)
 */

// Define version macro; evaluates to major * 1000 + minor * 10 + patch so that it's safe to use in less-than comparisons
// Note: pugixml used major * 100 + minor * 10 + patch format up until 1.9 (which had version identifier 190); starting from pugixml 1.10, the minor version number is two digits
#ifndef PUGIXML_VERSION
#	define PUGIXML_VERSION 1130 // 1.13
#endif

// Include user configuration file (this can define various configuration macros)
#include "pugiconfig.hpp"

#ifndef HEADER_PUGIXML_HPP
#define HEADER_PUGIXML_HPP

// Include stddef.h for size_t and ptrdiff_t
#include <stddef.h>

// Include exception header for XPath
#if !defined(PUGIXML_NO_XPATH) && !defined(PUGIXML_NO_EXCEPTIONS)
#	include <exception>
#endif

// Include STL headers
#ifndef PUGIXML_NO_STL
#	include <iterator>
#	include <iosfwd>
#	include <string>
#endif

// Macro for deprecated features
#ifndef PUGIXML_DEPRECATED
#	if defined(__GNUC__)
#		define PUGIXML_DEPRECATED __attribute__((deprecated))
#	elif defined(_MSC_VER) && _MSC_VER >= 1300
#		define PUGIXML_DEPRECATED __declspec(deprecated)
#	else
#		define PUGIXML_DEPRECATED
#	endif
#endif

// If no API is defined, assume default
#ifndef PUGIXML_API
#	define PUGIXML_API
#endif

// If no API for classes is defined, assume default
#ifndef PUGIXML_CLASS
#	define PUGIXML_CLASS PUGIXML_API
#endif

// If no API for functions is defined, assume default
#ifndef PUGIXML_FUNCTION
#	define PUGIXML_FUNCTION PUGIXML_API
#endif

// If the platform is known to have long long support, enable long long functions
#ifndef PUGIXML_HAS_LONG_LONG
#	if __cplusplus >= 201103
#		define PUGIXML_HAS_LONG_LONG
#	elif defined(_MSC_VER) && _MSC_VER >= 1400
#		define PUGIXML_HAS_LONG_LONG
#	endif
#endif

// If the platform is known to have move semantics support, compile move ctor/operator implementation
#ifndef PUGIXML_HAS_MOVE
#	if __cplusplus >= 201103
#		define PUGIXML_HAS_MOVE
#	elif defined(_MSC_VER) && _MSC_VER >= 1600
#		define PUGIXML_HAS_MOVE
#	endif
#endif

// If C++ is 2011 or higher, add 'noexcept' specifiers
#ifndef PUGIXML_NOEXCEPT
#	if __cplusplus >= 201103
#		define PUGIXML_NOEXCEPT noexcept
#	elif defined(_MSC_VER) && _MSC_VER >= 1900
#		define PUGIXML_NOEXCEPT noexcept
#	else
#		define PUGIXML_NOEXCEPT
#	endif
#endif

// Some functions can not be noexcept in compact mode
#ifdef PUGIXML_COMPACT
#	define PUGIXML_NOEXCEPT_IF_NOT_COMPACT
#else
#	define PUGIXML_NOEXCEPT_IF_NOT_COMPACT PUGIXML_NOEXCEPT
#endif

// If C++ is 2011 or higher, add 'override' qualifiers
#ifndef PUGIXML_OVERRIDE
#	if __cplusplus >= 201103
#		define PUGIXML_OVERRIDE override
#	elif defined(_MSC_VER) && _MSC_VER >= 1700
#		define PUGIXML_OVERRIDE override
#	else
#		define PUGIXML_OVERRIDE
#	endif
#endif

// If C++ is 2011 or higher, use 'nullptr'
#ifndef PUGIXML_NULL
#	if __cplusplus >= 201103
#		define PUGIXML_NULL nullptr
#	elif defined(_MSC_VER) && _MSC_VER >= 1600
#		define PUGIXML_NULL nullptr
#	else
#		define PUGIXML_NULL 0
#	endif
#endif

// Character interface macros
#ifdef PUGIXML_WCHAR_MODE
#	define PUGIXML_TEXT(t) L ## t
#	define PUGIXML_CHAR wchar_t
#else
#	define PUGIXML_TEXT(t) t
#	define PUGIXML_CHAR char
#endif

namespace pugi
{
	// Character type used for all internal storage and operations; depends on PUGIXML_WCHAR_MODE
	typedef PUGIXML_CHAR char_t;

#ifndef PUGIXML_NO_STL
	// String type used for operations that work with STL string; depends on PUGIXML_WCHAR_MODE
	typedef std::basic_string<PUGIXML_CHAR, std::char_traits<PUGIXML_CHAR>, std::allocator<PUGIXML_CHAR> > string_t;
#endif
}

// The PugiXML namespace
namespace pugi
{
	// Tree node types
	enum xml_node_type
	{
		node_null,			// Empty (null) node handle
		node_document,		// A document tree's absolute root
		node_element,		// Element tag, i.e. '<node/>'
		node_pcdata,		// Plain character data, i.e. 'text'
		node_cdata,			// Character data, i.e. '<![CDATA[text]]>'
		node_comment,		// Comment tag, i.e. '<!-- text -->'
		node_pi,			// Processing instruction, i.e. '<?name?>'
		node_declaration,	// Document declaration, i.e. '<?xml version="1.0"?>'
		node_doctype		// Document type declaration, i.e. '<!DOCTYPE doc>'
	};

	// Parsing options

	// Minimal parsing mode (equivalent to turning all other flags off).
	// Only elements and PCDATA sections are added to the DOM tree, no text conversions are performed.
	const unsigned int parse_minimal = 0x0000;

	// This flag determines if processing instructions (node_pi) are added to the DOM tree. This flag is off by default.
	const unsigned int parse_pi = 0x0001;

	// This flag determines if comments (node_comment) are added to the DOM tree. This flag is off by default.
	const unsigned int parse_comments = 0x0002;

	// This flag determines if CDATA sections (node_cdata) are added to the DOM tree. This flag is on by default.
	const unsigned int parse_cdata = 0x0004;

	// This flag determines if plain character data (node_pcdata) that consist only of whitespace are added to the DOM tree.
	// This flag is off by default; turning it on usually results in slower parsing and more memory consumption.
	const unsigned int parse_ws_pcdata = 0x0008;

	// This flag determines if character and entity references are expanded during parsing. This flag is on by default.
	const unsigned int parse_escapes = 0x0010;

	// This flag determines if EOL characters are normalized (converted to #xA) during parsing. This flag is on by default.
	const unsigned int parse_eol = 0x0020;

	// This flag determines if attribute values are normalized using CDATA normalization rules during parsing. This flag is on by default.
	const unsigned int parse_wconv_attribute = 0x0040;

	// This flag determines if attribute values are normalized using NMTOKENS normalization rules during parsing. This flag is off by default.
	const unsigned int parse_wnorm_attribute = 0x0080;

	// This flag determines if document declaration (node_declaration) is added to the DOM tree. This flag is off by default.
	const unsigned int parse_declaration = 0x0100;

	// This flag determines if document type declaration (node_doctype) is added to the DOM tree. This flag is off by default.
	const unsigned int parse_doctype = 0x0200;

	// This flag determines if plain character data (node_pcdata) that is the only child of the parent node and that consists only
	// of whitespace is added to the DOM tree.
	// This flag is off by default; turning it on may result in slower parsing and more memory consumption.
	const unsigned int parse_ws_pcdata_single = 0x0400;

	// This flag determines if leading and trailing whitespace is to be removed from plain character data. This flag is off by default.
	const unsigned int parse_trim_pcdata = 0x0800;

	// This flag determines if plain character data that does not have a parent node is added to the DOM tree, and if an empty document
	// is a valid document. This flag is off by default.
	const unsigned int parse_fragment = 0x1000;

	// This flag determines if plain character data is be stored in the parent element's value. This significantly changes the structure of
	// the document; this flag is only recommended for parsing documents with many PCDATA nodes in memory-constrained environments.
	// This flag is off by default.
	const unsigned int parse_embed_pcdata = 0x2000;

	// The default parsing mode.
	// Elements, PCDATA and CDATA sections are added to the DOM tree, character/reference entities are expanded,
	// End-of-Line characters are normalized, attribute values are normalized using CDATA normalization rules.
	const unsigned int parse_default = parse_cdata | parse_escapes | parse_wconv_attribute | parse_eol;

	// The full parsing mode.
	// Nodes of all types are added to the DOM tree, character/reference entities are expanded,
	// End-of-Line characters are normalized, attribute values are normalized using CDATA normalization rules.
	const unsigned int parse_full = parse_default | parse_pi | parse_comments | parse_declaration | parse_doctype;

	// These flags determine the encoding of input data for XML document
	enum xml_encoding
	{
		encoding_auto,		// Auto-detect input encoding using BOM or < / <? detection; use UTF8 if BOM is not found
		encoding_utf8,		// UTF8 encoding
		encoding_utf16_le,	// Little-endian UTF16
		encoding_utf16_be,	// Big-endian UTF16
		encoding_utf16,		// UTF16 with native endianness
		encoding_utf32_le,	// Little-endian UTF32
		encoding_utf32_be,	// Big-endian UTF32
		encoding_utf32,		// UTF32 with native endianness
		encoding_wchar,		// The same encoding wchar_t has (either UTF16 or UTF32)
		encoding_latin1
	};

	// Formatting flags

	// Indent the nodes that are written to output stream with as many indentation strings as deep the node is in DOM tree. This flag is on by default.
	const unsigned int format_indent = 0x01;

	// Write encoding-specific BOM to the output stream. This flag is off by default.
	const unsigned int format_write_bom = 0x02;

	// Use raw output mode (no indentation and no line breaks are written). This flag is off by default.
	const unsigned int format_raw = 0x04;

	// Omit default XML declaration even if there is no declaration in the document. This flag is off by default.
	const unsigned int format_no_declaration = 0x08;

	// Don't escape attribute values and PCDATA contents. This flag is off by default.
	const unsigned int format_no_escapes = 0x10;

	// Open file using text mode in xml_document::save_file. This enables special character (i.e. new-line) conversions on some systems. This flag is off by default.
	const unsigned int format_save_file_text = 0x20;

	// Write every attribute on a new line with appropriate indentation. This flag is off by default.
	const unsigned int format_indent_attributes = 0x40;

	// Don't output empty element tags, instead writing an explicit start and end tag even if there are no children. This flag is off by default.
	const unsigned int format_no_empty_element_tags = 0x80;

	// Skip characters belonging to range [0; 32) instead of "&#xNN;" encoding. This flag is off by default.
	const unsigned int format_skip_control_chars = 0x100;

	// Use single quotes ' instead of double quotes " for enclosing attribute values. This flag is off by default.
	const unsigned int format_attribute_single_quote = 0x200;

	// The default set of formatting flags.
	// Nodes are indented depending on their depth in DOM tree, a default declaration is output if document has none.
	const unsigned int format_default = format_indent;

	const int default_double_precision = 17;
	const int default_float_precision = 9;

	// Forward declarations
	struct xml_attribute_struct;
	struct xml_node_struct;

	class xml_node_iterator;
	class xml_attribute_iterator;
	class xml_named_node_iterator;

	class xml_tree_walker;

	struct xml_parse_result;

	class xml_node;

	class xml_text;

	#ifndef PUGIXML_NO_XPATH
	class xpath_node;
	class xpath_node_set;
	class xpath_query;
	class xpath_variable_set;
	#endif

	// Range-based for loop support
	template <typename It> class xml_object_range
	{
	public:
		typedef It const_iterator;
		typedef It iterator;

		xml_object_range(It b, It e): _begin(b), _end(e)
		{
		}

		It begin() const { return _begin; }
		It end() const { return _end; }

		bool empty() const { return _begin == _end; }

	private:
		It _begin, _end;
	};

	// Writer interface for node printing (see xml_node::print)
	class PUGIXML_CLASS xml_writer
	{
	public:
		virtual ~xml_writer();

		// Write memory chunk into stream/file/whatever
		virtual void write(const void* data, size_t size) = 0;
	};

	// xml_writer implementation for FILE*
	class PUGIXML_CLASS xml_writer_file: public xml_writer
	{
	public:
		// Construct writer from a FILE* object; void* is used to avoid header dependencies on stdio
		xml_writer_file(void* file);

		virtual void write(const void* data, size_t size) PUGIXML_OVERRIDE;

	private:
		void* file;
	};

	#ifndef PUGIXML_NO_STL
	// xml_writer implementation for streams
	class PUGIXML_CLASS xml_writer_stream: public xml_writer
	{
	public:
		// Construct writer from an output stream object
		xml_writer_stream(std::basic_ostream<char, std::char_traits<char> >& stream);
		xml_writer_stream(std::basic_ostream<wchar_t, std::char_traits<wchar_t> >& stream);

		virtual void write(const void* data, size_t size) PUGIXML_OVERRIDE;

	private:
		std::basic_ostream<char, std::char_traits<char> >* narrow_stream;
		std::basic_ostream<wchar_t, std::char_traits<wchar_t> >* wide_stream;
	};
	#endif

	// A light-weight handle for manipulating attributes in DOM tree
	class PUGIXML_CLASS xml_attribute
	{
		friend class xml_attribute_iterator;
		friend class xml_node;

	private:
		xml_attribute_struct* _attr;

		typedef void (*unspecified_bool_type)(xml_attribute***);

	public:
		// Default constructor. Constructs an empty attribute.
		xml_attribute();

		// Constructs attribute from internal pointer
		explicit xml_attribute(xml_attribute_struct* attr);

		// Safe bool conversion operator
		operator unspecified_bool_type() const;

		// Borland C++ workaround
		bool operator!() const;

		// Comparison operators (compares wrapped attribute pointers)
		bool operator==(const xml_attribute& r) const;
		bool operator!=(const xml_attribute& r) const;
		bool operator<(const xml_attribute& r) const;
		bool operator>(const xml_attribute& r) const;
		bool operator<=(const xml_attribute& r) const;
		bool operator>=(const xml_attribute& r) const;

		// Check if attribute is empty
		bool empty() const;

		// Get attribute name/value, or "" if attribute is empty
		const char_t* name() const;
		const char_t* value() const;

		// Get attribute value, or the default value if attribute is empty
		const char_t* as_string(const char_t* def = PUGIXML_TEXT("")) const;

		// Get attribute value as a number, or the default value if conversion did not succeed or attribute is empty
		int as_int(int def = 0) const;
		unsigned int as_uint(unsigned int def = 0) const;
		double as_double(double def = 0) const;
		float as_float(float def = 0) const;

	#ifdef PUGIXML_HAS_LONG_LONG
		long long as_llong(long long def = 0) const;
		unsigned long long as_ullong(unsigned long long def = 0) const;
	#endif

		// Get attribute value as bool (returns true if first character is in '1tTyY' set), or the default value if attribute is empty
		bool as_bool(bool def = false) const;

		// Set attribute name/value (returns false if attribute is empty or there is not enough memory)
		bool set_name(const char_t* rhs);
		bool set_value(const char_t* rhs, size_t sz);
		bool set_value(const char_t* rhs);

		// Set attribute value with type conversion (numbers are converted to strings, boolean is converted to "true"/"false")
		bool set_value(int rhs);
		bool set_value(unsigned int rhs);
		bool set_value(long rhs);
		bool set_value(unsigned long rhs);
		bool set_value(double rhs);
		bool set_value(double rhs, int precision);
		bool set_value(float rhs);
		bool set_value(float rhs, int precision);
		bool set_value(bool rhs);

	#ifdef PUGIXML_HAS_LONG_LONG
		bool set_value(long long rhs);
		bool set_value(unsigned long long rhs);
	#endif

		// Set attribute value (equivalent to set_value without error checking)
		xml_attribute& operator=(const char_t* rhs);
		xml_attribute& operator=(int rhs);
		xml_attribute& operator=(unsigned int rhs);
		xml_attribute& operator=(long rhs);
		xml_attribute& operator=(unsigned long rhs);
		xml_attribute& operator=(double rhs);
		xml_attribute& operator=(float rhs);
		xml_attribute& operator=(bool rhs);

	#ifdef PUGIXML_HAS_LONG_LONG
		xml_attribute& operator=(long long rhs);
		xml_attribute& operator=(unsigned long long rhs);
	#endif

		// Get next/previous attribute in the attribute list of the parent node
		xml_attribute next_attribute() const;
		xml_attribute previous_attribute() const;

		// Get hash value (unique for handles to the same object)
		size_t hash_value() const;

		// Get internal pointer
		xml_attribute_struct* internal_object() const;
	};

#ifdef __BORLANDC__
	// Borland C++ workaround
	bool PUGIXML_FUNCTION operator&&(const xml_attribute& lhs, bool rhs);
	bool PUGIXML_FUNCTION operator||(const xml_attribute& lhs, bool rhs);
#endif

	// A light-weight handle for manipulating nodes in DOM tree
	class PUGIXML_CLASS xml_node
	{
		friend class xml_attribute_iterator;
		friend class xml_node_iterator;
		friend class xml_named_node_iterator;

	protected:
		xml_node_struct* _root;

		typedef void (*unspecified_bool_type)(xml_node***);

	public:
		// Default constructor. Constructs an empty node.
		xml_node();

		// Constructs node from internal pointer
		explicit xml_node(xml_node_struct* p);

		// Safe bool conversion operator
		operator unspecified_bool_type() const;

		// Borland C++ workaround
		bool operator!() const;

		// Comparison operators (compares wrapped node pointers)
		bool operator==(const xml_node& r) const;
		bool operator!=(const xml_node& r) const;
		bool operator<(const xml_node& r) const;
		bool operator>(const xml_node& r) const;
		bool operator<=(const xml_node& r) const;
		bool operator>=(const xml_node& r) const;

		// Check if node is empty.
		bool empty() const;

		// Get node type
		xml_node_type type() const;

		// Get node name, or "" if node is empty or it has no name
		const char_t* name() const;

		// Get node value, or "" if node is empty or it has no value
		// Note: For <node>text</node> node.value() does not return "text"! Use child_value() or text() methods to access text inside nodes.
		const char_t* value() const;

		// Get attribute list
		xml_attribute first_attribute() const;
		xml_attribute last_attribute() const;

		// Get children list
		xml_node first_child() const;
		xml_node last_child() const;

		// Get next/previous sibling in the children list of the parent node
		xml_node next_sibling() const;
		xml_node previous_sibling() const;

		// Get parent node
		xml_node parent() const;

		// Get root of DOM tree this node belongs to
		xml_node root() const;

		// Get text object for the current node
		xml_text text() const;

		// Get child, attribute or next/previous sibling with the specified name
		xml_node child(const char_t* name) const;
		xml_attribute attribute(const char_t* name) const;
		xml_node next_sibling(const char_t* name) const;
		xml_node previous_sibling(const char_t* name) const;

		// Get attribute, starting the search from a hint (and updating hint so that searching for a sequence of attributes is fast)
		xml_attribute attribute(const char_t* name, xml_attribute& hint) const;

		// Get child value of current node; that is, value of the first child node of type PCDATA/CDATA
		const char_t* child_value() const;

		// Get child value of child with specified name. Equivalent to child(name).child_value().
		const char_t* child_value(const char_t* name) const;

		// Set node name/value (returns false if node is empty, there is not enough memory, or node can not have name/value)
		bool set_name(const char_t* rhs);
		bool set_value(const char_t* rhs, size_t sz);
		bool set_value(const char_t* rhs);

		// Add attribute with specified name. Returns added attribute, or empty attribute on errors.
		xml_attribute append_attribute(const char_t* name);
		xml_attribute prepend_attribute(const char_t* name);
		xml_attribute insert_attribute_after(const char_t* name, const xml_attribute& attr);
		xml_attribute insert_attribute_before(const char_t* name, const xml_attribute& attr);

		// Add a copy of the specified attribute. Returns added attribute, or empty attribute on errors.
		xml_attribute append_copy(const xml_attribute& proto);
		xml_attribute prepend_copy(const xml_attribute& proto);
		xml_attribute insert_copy_after(const xml_attribute& proto, const xml_attribute& attr);
		xml_attribute insert_copy_before(const xml_attribute& proto, const xml_attribute& attr);

		// Add child node with specified type. Returns added node, or empty node on errors.
		xml_node append_child(xml_node_type type = node_element);
		xml_node prepend_child(xml_node_type type = node_element);
		xml_node insert_child_after(xml_node_type type, const xml_node& node);
		xml_node insert_child_before(xml_node_type type, const xml_node& node);

		// Add child element with specified name. Returns added node, or empty node on errors.
		xml_node append_child(const char_t* name);
		xml_node prepend_child(const char_t* name);
		xml_node insert_child_after(const char_t* name, const xml_node& node);
		xml_node insert_child_before(const char_t* name, const xml_node& node);

		// Add a copy of the specified node as a child. Returns added node, or empty node on errors.
		xml_node append_copy(const xml_node& proto);
		xml_node prepend_copy(const xml_node& proto);
		xml_node insert_copy_after(const xml_node& proto, const xml_node& node);
		xml_node insert_copy_before(const xml_node& proto, const xml_node& node);

		// Move the specified node to become a child of this node. Returns moved node, or empty node on errors.
		xml_node append_move(const xml_node& moved);
		xml_node prepend_move(const xml_node& moved);
		xml_node insert_move_after(const xml_node& moved, const xml_node& node);
		xml_node insert_move_before(const xml_node& moved, const xml_node& node);

		// Remove specified attribute
		bool remove_attribute(const xml_attribute& a);
		bool remove_attribute(const char_t* name);

		// Remove all attributes
		bool remove_attributes();

		// Remove specified child
		bool remove_child(const xml_node& n);
		bool remove_child(const char_t* name);

		// Remove all children
		bool remove_children();

		// Parses buffer as an XML document fragment and appends all nodes as children of the current node.
		// Copies/converts the buffer, so it may be deleted or changed after the function returns.
		// Note: append_buffer allocates memory that has the lifetime of the owning document; removing the appended nodes does not immediately reclaim that memory.
		xml_parse_result append_buffer(const void* contents, size_t size, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);

		// Find attribute using predicate. Returns first attribute for which predicate returned true.
		template <typename Predicate> xml_attribute find_attribute(Predicate pred) const
		{
			if (!_root) return xml_attribute();

			for (xml_attribute attrib = first_attribute(); attrib; attrib = attrib.next_attribute())
				if (pred(attrib))
					return attrib;

			return xml_attribute();
		}

		// Find child node using predicate. Returns first child for which predicate returned true.
		template <typename Predicate> xml_node find_child(Predicate pred) const
		{
			if (!_root) return xml_node();

			for (xml_node node = first_child(); node; node = node.next_sibling())
				if (pred(node))
					return node;

			return xml_node();
		}

		// Find node from subtree using predicate. Returns first node from subtree (depth-first), for which predicate returned true.
		template <typename Predicate> xml_node find_node(Predicate pred) const
		{
			if (!_root) return xml_node();

			xml_node cur = first_child();

			while (cur._root && cur._root != _root)
			{
				if (pred(cur)) return cur;

				if (cur.first_child()) cur = cur.first_child();
				else if (cur.next_sibling()) cur = cur.next_sibling();
				else
				{
					while (!cur.next_sibling() && cur._root != _root) cur = cur.parent();

					if (cur._root != _root) cur = cur.next_sibling();
				}
			}

			return xml_node();
		}

		// Find child node by attribute name/value
		xml_node find_child_by_attribute(const char_t* name, const char_t* attr_name, const char_t* attr_value) const;
		xml_node find_child_by_attribute(const char_t* attr_name, const char_t* attr_value) const;

	#ifndef PUGIXML_NO_STL
		// Get the absolute node path from root as a text string.
		string_t path(char_t delimiter = '/') const;
	#endif

		// Search for a node by path consisting of node names and . or .. elements.
		xml_node first_element_by_path(const char_t* path, char_t delimiter = '/') const;

		// Recursively traverse subtree with xml_tree_walker
		bool traverse(xml_tree_walker& walker);

	#ifndef PUGIXML_NO_XPATH
		// Select single node by evaluating XPath query. Returns first node from the resulting node set.
		xpath_node select_node(const char_t* query, xpath_variable_set* variables = PUGIXML_NULL) const;
		xpath_node select_node(const xpath_query& query) const;

		// Select node set by evaluating XPath query
		xpath_node_set select_nodes(const char_t* query, xpath_variable_set* variables = PUGIXML_NULL) const;
		xpath_node_set select_nodes(const xpath_query& query) const;

		// (deprecated: use select_node instead) Select single node by evaluating XPath query.
		PUGIXML_DEPRECATED xpath_node select_single_node(const char_t* query, xpath_variable_set* variables = PUGIXML_NULL) const;
		PUGIXML_DEPRECATED xpath_node select_single_node(const xpath_query& query) const;

	#endif

		// Print subtree using a writer object
		void print(xml_writer& writer, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, xml_encoding encoding = encoding_auto, unsigned int depth = 0) const;

	#ifndef PUGIXML_NO_STL
		// Print subtree to stream
		void print(std::basic_ostream<char, std::char_traits<char> >& os, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, xml_encoding encoding = encoding_auto, unsigned int depth = 0) const;
		void print(std::basic_ostream<wchar_t, std::char_traits<wchar_t> >& os, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, unsigned int depth = 0) const;
	#endif

		// Child nodes iterators
		typedef xml_node_iterator iterator;

		iterator begin() const;
		iterator end() const;

		// Attribute iterators
		typedef xml_attribute_iterator attribute_iterator;

		attribute_iterator attributes_begin() const;
		attribute_iterator attributes_end() const;

		// Range-based for support
		xml_object_range<xml_node_iterator> children() const;
		xml_object_range<xml_attribute_iterator> attributes() const;

		// Range-based for support for all children with the specified name
		// Note: name pointer must have a longer lifetime than the returned object; be careful with passing temporaries!
		xml_object_range<xml_named_node_iterator> children(const char_t* name) const;

		// Get node offset in parsed file/string (in char_t units) for debugging purposes
		ptrdiff_t offset_debug() const;

		// Get hash value (unique for handles to the same object)
		size_t hash_value() const;

		// Get internal pointer
		xml_node_struct* internal_object() const;
	};

#ifdef __BORLANDC__
	// Borland C++ workaround
	bool PUGIXML_FUNCTION operator&&(const xml_node& lhs, bool rhs);
	bool PUGIXML_FUNCTION operator||(const xml_node& lhs, bool rhs);
#endif

	// A helper for working with text inside PCDATA nodes
	class PUGIXML_CLASS xml_text
	{
		friend class xml_node;

		xml_node_struct* _root;

		typedef void (*unspecified_bool_type)(xml_text***);

		explicit xml_text(xml_node_struct* root);

		xml_node_struct* _data_new();
		xml_node_struct* _data() const;

	public:
		// Default constructor. Constructs an empty object.
		xml_text();

		// Safe bool conversion operator
		operator unspecified_bool_type() const;

		// Borland C++ workaround
		bool operator!() const;

		// Check if text object is empty
		bool empty() const;

		// Get text, or "" if object is empty
		const char_t* get() const;

		// Get text, or the default value if object is empty
		const char_t* as_string(const char_t* def = PUGIXML_TEXT("")) const;

		// Get text as a number, or the default value if conversion did not succeed or object is empty
		int as_int(int def = 0) const;
		unsigned int as_uint(unsigned int def = 0) const;
		double as_double(double def = 0) const;
		float as_float(float def = 0) const;

	#ifdef PUGIXML_HAS_LONG_LONG
		long long as_llong(long long def = 0) const;
		unsigned long long as_ullong(unsigned long long def = 0) const;
	#endif

		// Get text as bool (returns true if first character is in '1tTyY' set), or the default value if object is empty
		bool as_bool(bool def = false) const;

		// Set text (returns false if object is empty or there is not enough memory)
		bool set(const char_t* rhs, size_t sz);
		bool set(const char_t* rhs);

		// Set text with type conversion (numbers are converted to strings, boolean is converted to "true"/"false")
		bool set(int rhs);
		bool set(unsigned int rhs);
		bool set(long rhs);
		bool set(unsigned long rhs);
		bool set(double rhs);
		bool set(double rhs, int precision);
		bool set(float rhs);
		bool set(float rhs, int precision);
		bool set(bool rhs);

	#ifdef PUGIXML_HAS_LONG_LONG
		bool set(long long rhs);
		bool set(unsigned long long rhs);
	#endif

		// Set text (equivalent to set without error checking)
		xml_text& operator=(const char_t* rhs);
		xml_text& operator=(int rhs);
		xml_text& operator=(unsigned int rhs);
		xml_text& operator=(long rhs);
		xml_text& operator=(unsigned long rhs);
		xml_text& operator=(double rhs);
		xml_text& operator=(float rhs);
		xml_text& operator=(bool rhs);

	#ifdef PUGIXML_HAS_LONG_LONG
		xml_text& operator=(long long rhs);
		xml_text& operator=(unsigned long long rhs);
	#endif

		// Get the data node (node_pcdata or node_cdata) for this object
		xml_node data() const;
	};

#ifdef __BORLANDC__
	// Borland C++ workaround
	bool PUGIXML_FUNCTION operator&&(const xml_text& lhs, bool rhs);
	bool PUGIXML_FUNCTION operator||(const xml_text& lhs, bool rhs);
#endif

	// Child node iterator (a bidirectional iterator over a collection of xml_node)
	class PUGIXML_CLASS xml_node_iterator
	{
		friend class xml_node;

	private:
		mutable xml_node _wrap;
		xml_node _parent;

		xml_node_iterator(xml_node_struct* ref, xml_node_struct* parent);

	public:
		// Iterator traits
		typedef ptrdiff_t difference_type;
		typedef xml_node value_type;
		typedef xml_node* pointer;
		typedef xml_node& reference;

	#ifndef PUGIXML_NO_STL
		typedef std::bidirectional_iterator_tag iterator_category;
	#endif

		// Default constructor
		xml_node_iterator();

		// Construct an iterator which points to the specified node
		xml_node_iterator(const xml_node& node);

		// Iterator operators
		bool operator==(const xml_node_iterator& rhs) const;
		bool operator!=(const xml_node_iterator& rhs) const;

		xml_node& operator*() const;
		xml_node* operator->() const;

		xml_node_iterator& operator++();
		xml_node_iterator operator++(int);

		xml_node_iterator& operator--();
		xml_node_iterator operator--(int);
	};

	// Attribute iterator (a bidirectional iterator over a collection of xml_attribute)
	class PUGIXML_CLASS xml_attribute_iterator
	{
		friend class xml_node;

	private:
		mutable xml_attribute _wrap;
		xml_node _parent;

		xml_attribute_iterator(xml_attribute_struct* ref, xml_node_struct* parent);

	public:
		// Iterator traits
		typedef ptrdiff_t difference_type;
		typedef xml_attribute value_type;
		typedef xml_attribute* pointer;
		typedef xml_attribute& reference;

	#ifndef PUGIXML_NO_STL
		typedef std::bidirectional_iterator_tag iterator_category;
	#endif

		// Default constructor
		xml_attribute_iterator();

		// Construct an iterator which points to the specified attribute
		xml_attribute_iterator(const xml_attribute& attr, const xml_node& parent);

		// Iterator operators
		bool operator==(const xml_attribute_iterator& rhs) const;
		bool operator!=(const xml_attribute_iterator& rhs) const;

		xml_attribute& operator*() const;
		xml_attribute* operator->() const;

		xml_attribute_iterator& operator++();
		xml_attribute_iterator operator++(int);

		xml_attribute_iterator& operator--();
		xml_attribute_iterator operator--(int);
	};

	// Named node range helper
	class PUGIXML_CLASS xml_named_node_iterator
	{
		friend class xml_node;

	public:
		// Iterator traits
		typedef ptrdiff_t difference_type;
		typedef xml_node value_type;
		typedef xml_node* pointer;
		typedef xml_node& reference;

	#ifndef PUGIXML_NO_STL
		typedef std::bidirectional_iterator_tag iterator_category;
	#endif

		// Default constructor
		xml_named_node_iterator();

		// Construct an iterator which points to the specified node
		// Note: name pointer is stored in the iterator and must have a longer lifetime than iterator itself
		xml_named_node_iterator(const xml_node& node, const char_t* name);

		// Iterator operators
		bool operator==(const xml_named_node_iterator& rhs) const;
		bool operator!=(const xml_named_node_iterator& rhs) const;

		xml_node& operator*() const;
		xml_node* operator->() const;

		xml_named_node_iterator& operator++();
		xml_named_node_iterator operator++(int);

		xml_named_node_iterator& operator--();
		xml_named_node_iterator operator--(int);

	private:
		mutable xml_node _wrap;
		xml_node _parent;
		const char_t* _name;

		xml_named_node_iterator(xml_node_struct* ref, xml_node_struct* parent, const char_t* name);
	};

	// Abstract tree walker class (see xml_node::traverse)
	class PUGIXML_CLASS xml_tree_walker
	{
		friend class xml_node;

	private:
		int _depth;

	protected:
		// Get current traversal depth
		int depth() const;

	public:
		xml_tree_walker();
		virtual ~xml_tree_walker();

		// Callback that is called when traversal begins
		virtual bool begin(xml_node& node);

		// Callback that is called for each node traversed
		virtual bool for_each(xml_node& node) = 0;

		// Callback that is called when traversal ends
		virtual bool end(xml_node& node);
	};

	// Parsing status, returned as part of xml_parse_result object
	enum xml_parse_status
	{
		status_ok = 0,				// No error

		status_file_not_found,		// File was not found during load_file()
		status_io_error,			// Error reading from file/stream
		status_out_of_memory,		// Could not allocate memory
		status_internal_error,		// Internal error occurred

		status_unrecognized_tag,	// Parser could not determine tag type

		status_bad_pi,				// Parsing error occurred while parsing document declaration/processing instruction
		status_bad_comment,			// Parsing error occurred while parsing comment
		status_bad_cdata,			// Parsing error occurred while parsing CDATA section
		status_bad_doctype,			// Parsing error occurred while parsing document type declaration
		status_bad_pcdata,			// Parsing error occurred while parsing PCDATA section
		status_bad_start_element,	// Parsing error occurred while parsing start element tag
		status_bad_attribute,		// Parsing error occurred while parsing element attribute
		status_bad_end_element,		// Parsing error occurred while parsing end element tag
		status_end_element_mismatch,// There was a mismatch of start-end tags (closing tag had incorrect name, some tag was not closed or there was an excessive closing tag)

		status_append_invalid_root,	// Unable to append nodes since root type is not node_element or node_document (exclusive to xml_node::append_buffer)

		status_no_document_element	// Parsing resulted in a document without element nodes
	};

	// Parsing result
	struct PUGIXML_CLASS xml_parse_result
	{
		// Parsing status (see xml_parse_status)
		xml_parse_status status;

		// Last parsed offset (in char_t units from start of input data)
		ptrdiff_t offset;

		// Source document encoding
		xml_encoding encoding;

		// Default constructor, initializes object to failed state
		xml_parse_result();

		// Cast to bool operator
		operator bool() const;

		// Get error description
		const char* description() const;
	};

	// Document class (DOM tree root)
	class PUGIXML_CLASS xml_document: public xml_node
	{
	private:
		char_t* _buffer;

		char _memory[192];

		// Non-copyable semantics
		xml_document(const xml_document&);
		xml_document& operator=(const xml_document&);

		void _create();
		void _destroy();
		void _move(xml_document& rhs) PUGIXML_NOEXCEPT_IF_NOT_COMPACT;

	public:
		// Default constructor, makes empty document
		xml_document();

		// Destructor, invalidates all node/attribute handles to this document
		~xml_document();

	#ifdef PUGIXML_HAS_MOVE
		// Move semantics support
		xml_document(xml_document&& rhs) PUGIXML_NOEXCEPT_IF_NOT_COMPACT;
		xml_document& operator=(xml_document&& rhs) PUGIXML_NOEXCEPT_IF_NOT_COMPACT;
	#endif

		// Removes all nodes, leaving the empty document
		void reset();

		// Removes all nodes, then copies the entire contents of the specified document
		void reset(const xml_document& proto);

	#ifndef PUGIXML_NO_STL
		// Load document from stream.
		xml_parse_result load(std::basic_istream<char, std::char_traits<char> >& stream, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);
		xml_parse_result load(std::basic_istream<wchar_t, std::char_traits<wchar_t> >& stream, unsigned int options = parse_default);
	#endif

		// (deprecated: use load_string instead) Load document from zero-terminated string. No encoding conversions are applied.
		PUGIXML_DEPRECATED xml_parse_result load(const char_t* contents, unsigned int options = parse_default);

		// Load document from zero-terminated string. No encoding conversions are applied.
		xml_parse_result load_string(const char_t* contents, unsigned int options = parse_default);

		// Load document from file
		xml_parse_result load_file(const char* path, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);
		xml_parse_result load_file(const wchar_t* path, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);

		// Load document from buffer. Copies/converts the buffer, so it may be deleted or changed after the function returns.
		xml_parse_result load_buffer(const void* contents, size_t size, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);

		// Load document from buffer, using the buffer for in-place parsing (the buffer is modified and used for storage of document data).
		// You should ensure that buffer data will persist throughout the document's lifetime, and free the buffer memory manually once document is destroyed.
		xml_parse_result load_buffer_inplace(void* contents, size_t size, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);

		// Load document from buffer, using the buffer for in-place parsing (the buffer is modified and used for storage of document data).
		// You should allocate the buffer with pugixml allocation function; document will free the buffer when it is no longer needed (you can't use it anymore).
		xml_parse_result load_buffer_inplace_own(void* contents, size_t size, unsigned int options = parse_default, xml_encoding encoding = encoding_auto);

		// Save XML document to writer (semantics is slightly different from xml_node::print, see documentation for details).
		void save(xml_writer& writer, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, xml_encoding encoding = encoding_auto) const;

	#ifndef PUGIXML_NO_STL
		// Save XML document to stream (semantics is slightly different from xml_node::print, see documentation for details).
		void save(std::basic_ostream<char, std::char_traits<char> >& stream, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, xml_encoding encoding = encoding_auto) const;
		void save(std::basic_ostream<wchar_t, std::char_traits<wchar_t> >& stream, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default) const;
	#endif

		// Save XML to file
		bool save_file(const char* path, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, xml_encoding encoding = encoding_auto) const;
		bool save_file(const wchar_t* path, const char_t* indent = PUGIXML_TEXT("\t"), unsigned int flags = format_default, xml_encoding encoding = encoding_auto) const;

		// Get document element
		xml_node document_element() const;
	};

#ifndef PUGIXML_NO_XPATH
	// XPath query return type
	enum xpath_value_type
	{
		xpath_type_none,	  // Unknown type (query failed to compile)
		xpath_type_node_set,  // Node set (xpath_node_set)
		xpath_type_number,	  // Number
		xpath_type_string,	  // String
		xpath_type_boolean	  // Boolean
	};

	// XPath parsing result
	struct PUGIXML_CLASS xpath_parse_result
	{
		// Error message (0 if no error)
		const char* error;

		// Last parsed offset (in char_t units from string start)
		ptrdiff_t offset;

		// Default constructor, initializes object to failed state
		xpath_parse_result();

		// Cast to bool operator
		operator bool() const;

		// Get error description
		const char* description() const;
	};

	// A single XPath variable
	class PUGIXML_CLASS xpath_variable
	{
		friend class xpath_variable_set;

	protected:
		xpath_value_type _type;
		xpath_variable* _next;

		xpath_variable(xpath_value_type type);

		// Non-copyable semantics
		xpath_variable(const xpath_variable&);
		xpath_variable& operator=(const xpath_variable&);

	public:
		// Get variable name
		const char_t* name() const;

		// Get variable type
		xpath_value_type type() const;

		// Get variable value; no type conversion is performed, default value (false, NaN, empty string, empty node set) is returned on type mismatch error
		bool get_boolean() const;
		double get_number() const;
		const char_t* get_string() const;
		const xpath_node_set& get_node_set() const;

		// Set variable value; no type conversion is performed, false is returned on type mismatch error
		bool set(bool value);
		bool set(double value);
		bool set(const char_t* value);
		bool set(const xpath_node_set& value);
	};

	// A set of XPath variables
	class PUGIXML_CLASS xpath_variable_set
	{
	private:
		xpath_variable* _data[64];

		void _assign(const xpath_variable_set& rhs);
		void _swap(xpath_variable_set& rhs);

		xpath_variable* _find(const char_t* name) const;

		static bool _clone(xpath_variable* var, xpath_variable** out_result);
		static void _destroy(xpath_variable* var);

	public:
		// Default constructor/destructor
		xpath_variable_set();
		~xpath_variable_set();

		// Copy constructor/assignment operator
		xpath_variable_set(const xpath_variable_set& rhs);
		xpath_variable_set& operator=(const xpath_variable_set& rhs);

	#ifdef PUGIXML_HAS_MOVE
		// Move semantics support
		xpath_variable_set(xpath_variable_set&& rhs) PUGIXML_NOEXCEPT;
		xpath_variable_set& operator=(xpath_variable_set&& rhs) PUGIXML_NOEXCEPT;
	#endif

		// Add a new variable or get the existing one, if the types match
		xpath_variable* add(const char_t* name, xpath_value_type type);

		// Set value of an existing variable; no type conversion is performed, false is returned if there is no such variable or if types mismatch
		bool set(const char_t* name, bool value);
		bool set(const char_t* name, double value);
		bool set(const char_t* name, const char_t* value);
		bool set(const char_t* name, const xpath_node_set& value);

		// Get existing variable by name
		xpath_variable* get(const char_t* name);
		const xpath_variable* get(const char_t* name) const;
	};

	// A compiled XPath query object
	class PUGIXML_CLASS xpath_query
	{
	private:
		void* _impl;
		xpath_parse_result _result;

		typedef void (*unspecified_bool_type)(xpath_query***);

		// Non-copyable semantics
		xpath_query(const xpath_query&);
		xpath_query& operator=(const xpath_query&);

	public:
		// Construct a compiled object from XPath expression.
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws xpath_exception on compilation errors.
		explicit xpath_query(const char_t* query, xpath_variable_set* variables = PUGIXML_NULL);

		// Constructor
		xpath_query();

		// Destructor
		~xpath_query();

	#ifdef PUGIXML_HAS_MOVE
		// Move semantics support
		xpath_query(xpath_query&& rhs) PUGIXML_NOEXCEPT;
		xpath_query& operator=(xpath_query&& rhs) PUGIXML_NOEXCEPT;
	#endif

		// Get query expression return type
		xpath_value_type return_type() const;

		// Evaluate expression as boolean value in the specified context; performs type conversion if necessary.
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws std::bad_alloc on out of memory errors.
		bool evaluate_boolean(const xpath_node& n) const;

		// Evaluate expression as double value in the specified context; performs type conversion if necessary.
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws std::bad_alloc on out of memory errors.
		double evaluate_number(const xpath_node& n) const;

	#ifndef PUGIXML_NO_STL
		// Evaluate expression as string value in the specified context; performs type conversion if necessary.
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws std::bad_alloc on out of memory errors.
		string_t evaluate_string(const xpath_node& n) const;
	#endif

		// Evaluate expression as string value in the specified context; performs type conversion if necessary.
		// At most capacity characters are written to the destination buffer, full result size is returned (includes terminating zero).
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws std::bad_alloc on out of memory errors.
		// If PUGIXML_NO_EXCEPTIONS is defined, returns empty  set instead.
		size_t evaluate_string(char_t* buffer, size_t capacity, const xpath_node& n) const;

		// Evaluate expression as node set in the specified context.
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws xpath_exception on type mismatch and std::bad_alloc on out of memory errors.
		// If PUGIXML_NO_EXCEPTIONS is defined, returns empty node set instead.
		xpath_node_set evaluate_node_set(const xpath_node& n) const;

		// Evaluate expression as node set in the specified context.
		// Return first node in document order, or empty node if node set is empty.
		// If PUGIXML_NO_EXCEPTIONS is not defined, throws xpath_exception on type mismatch and std::bad_alloc on out of memory errors.
		// If PUGIXML_NO_EXCEPTIONS is defined, returns empty node instead.
		xpath_node evaluate_node(const xpath_node& n) const;

		// Get parsing result (used to get compilation errors in PUGIXML_NO_EXCEPTIONS mode)
		const xpath_parse_result& result() const;

		// Safe bool conversion operator
		operator unspecified_bool_type() const;

		// Borland C++ workaround
		bool operator!() const;
	};

	#ifndef PUGIXML_NO_EXCEPTIONS
        #if defined(_MSC_VER)
          // C4275 can be ignored in Visual C++ if you are deriving
          // from a type in the Standard C++ Library
          #pragma warning(push)
          #pragma warning(disable: 4275)
        #endif
	// XPath exception class
	class PUGIXML_CLASS xpath_exception: public std::exception
	{
	private:
		xpath_parse_result _result;

	public:
		// Construct exception from parse result
		explicit xpath_exception(const xpath_parse_result& result);

		// Get error message
		virtual const char* what() const throw() PUGIXML_OVERRIDE;

		// Get parse result
		const xpath_parse_result& result() const;
	};
        #if defined(_MSC_VER)
          #pragma warning(pop)
        #endif
	#endif

	// XPath node class (either xml_node or xml_attribute)
	class PUGIXML_CLASS xpath_node
	{
	private:
		xml_node _node;
		xml_attribute _attribute;

		typedef void (*unspecified_bool_type)(xpath_node***);

	public:
		// Default constructor; constructs empty XPath node
		xpath_node();

		// Construct XPath node from XML node/attribute
		xpath_node(const xml_node& node);
		xpath_node(const xml_attribute& attribute, const xml_node& parent);

		// Get node/attribute, if any
		xml_node node() const;
		xml_attribute attribute() const;

		// Get parent of contained node/attribute
		xml_node parent() const;

		// Safe bool conversion operator
		operator unspecified_bool_type() const;

		// Borland C++ workaround
		bool operator!() const;

		// Comparison operators
		bool operator==(const xpath_node& n) const;
		bool operator!=(const xpath_node& n) const;
	};

#ifdef __BORLANDC__
	// Borland C++ workaround
	bool PUGIXML_FUNCTION operator&&(const xpath_node& lhs, bool rhs);
	bool PUGIXML_FUNCTION operator||(const xpath_node& lhs, bool rhs);
#endif

	// A fixed-size collection of XPath nodes
	class PUGIXML_CLASS xpath_node_set
	{
	public:
		// Collection type
		enum type_t
		{
			type_unsorted,			// Not ordered
			type_sorted,			// Sorted by document order (ascending)
			type_sorted_reverse		// Sorted by document order (descending)
		};

		// Constant iterator type
		typedef const xpath_node* const_iterator;

		// We define non-constant iterator to be the same as constant iterator so that various generic algorithms (i.e. boost foreach) work
		typedef const xpath_node* iterator;

		// Default constructor. Constructs empty set.
		xpath_node_set();

		// Constructs a set from iterator range; data is not checked for duplicates and is not sorted according to provided type, so be careful
		xpath_node_set(const_iterator begin, const_iterator end, type_t type = type_unsorted);

		// Destructor
		~xpath_node_set();

		// Copy constructor/assignment operator
		xpath_node_set(const xpath_node_set& ns);
		xpath_node_set& operator=(const xpath_node_set& ns);

	#ifdef PUGIXML_HAS_MOVE
		// Move semantics support
		xpath_node_set(xpath_node_set&& rhs) PUGIXML_NOEXCEPT;
		xpath_node_set& operator=(xpath_node_set&& rhs) PUGIXML_NOEXCEPT;
	#endif

		// Get collection type
		type_t type() const;

		// Get collection size
		size_t size() const;

		// Indexing operator
		const xpath_node& operator[](size_t index) const;

		// Collection iterators
		const_iterator begin() const;
		const_iterator end() const;

		// Sort the collection in ascending/descending order by document order
		void sort(bool reverse = false);

		// Get first node in the collection by document order
		xpath_node first() const;

		// Check if collection is empty
		bool empty() const;

	private:
		type_t _type;

		xpath_node _storage[1];

		xpath_node* _begin;
		xpath_node* _end;

		void _assign(const_iterator begin, const_iterator end, type_t type);
		void _move(xpath_node_set& rhs) PUGIXML_NOEXCEPT;
	};
#endif

#ifndef PUGIXML_NO_STL
	// Convert wide string to UTF8
	std::basic_string<char, std::char_traits<char>, std::allocator<char> > PUGIXML_FUNCTION as_utf8(const wchar_t* str);
	std::basic_string<char, std::char_traits<char>, std::allocator<char> > PUGIXML_FUNCTION as_utf8(const std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >& str);

	// Convert UTF8 to wide string
	std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > PUGIXML_FUNCTION as_wide(const char* str);
	std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> > PUGIXML_FUNCTION as_wide(const std::basic_string<char, std::char_traits<char>, std::allocator<char> >& str);
#endif

	// Memory allocation function interface; returns pointer to allocated memory or NULL on failure
	typedef void* (*allocation_function)(size_t size);

	// Memory deallocation function interface
	typedef void (*deallocation_function)(void* ptr);

	// Override default memory management functions. All subsequent allocations/deallocations will be performed via supplied functions.
	void PUGIXML_FUNCTION set_memory_management_functions(allocation_function allocate, deallocation_function deallocate);

	// Get current memory management functions
	allocation_function PUGIXML_FUNCTION get_memory_allocation_function();
	deallocation_function PUGIXML_FUNCTION get_memory_deallocation_function();
}

#if !defined(PUGIXML_NO_STL) && (defined(_MSC_VER) || defined(__ICC))
namespace std
{
	// Workarounds for (non-standard) iterator category detection for older versions (MSVC7/IC8 and earlier)
	std::bidirectional_iterator_tag PUGIXML_FUNCTION _Iter_cat(const pugi::xml_node_iterator&);
	std::bidirectional_iterator_tag PUGIXML_FUNCTION _Iter_cat(const pugi::xml_attribute_iterator&);
	std::bidirectional_iterator_tag PUGIXML_FUNCTION _Iter_cat(const pugi::xml_named_node_iterator&);
}
#endif

#if !defined(PUGIXML_NO_STL) && defined(__SUNPRO_CC)
namespace std
{
	// Workarounds for (non-standard) iterator category detection
	std::bidirectional_iterator_tag PUGIXML_FUNCTION __iterator_category(const pugi::xml_node_iterator&);
	std::bidirectional_iterator_tag PUGIXML_FUNCTION __iterator_category(const pugi::xml_attribute_iterator&);
	std::bidirectional_iterator_tag PUGIXML_FUNCTION __iterator_category(const pugi::xml_named_node_iterator&);
}
#endif

#endif

// Make sure implementation is included in header-only mode
// Use macro expansion in #include to work around QMake (QTBUG-11923)
#if defined(PUGIXML_HEADER_ONLY) && !defined(PUGIXML_SOURCE)
#	define PUGIXML_SOURCE "pugixml.cpp"
#	include PUGIXML_SOURCE
#endif

/**
 * Copyright (c) 2006-2022 Arseny Kapoulkine
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
