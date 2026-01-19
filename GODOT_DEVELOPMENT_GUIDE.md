# 🎯 Godot Engine Development Guide
## The Complete Guide to Extending GDScript

**Based on real-world struct implementation experience**  
**Version:** 1.0 - 2026-01-19

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Adding New Language Features](#adding-new-language-features)
3. [Parser Deep Dive](#parser-deep-dive)
4. [Type System](#type-system)
5. [Build System](#build-system)
6. [Testing Patterns](#testing-patterns)
7. [Common Gotchas](#common-gotchas)
8. [Code Patterns](#code-patterns)

---

## Architecture Overview

### GDScript Pipeline

```
Source Code
    ↓
Tokenizer (gdscript_tokenizer.cpp)
    ↓ (Token stream)
Parser (gdscript_parser.cpp)
    ↓ (AST - Abstract Syntax Tree)
Analyzer (gdscript_analyzer.cpp)
    ↓ (Type-checked AST)
Compiler (gdscript_compiler.cpp)
    ↓ (Bytecode)
VM (gdscript_vm.cpp)
    ↓ (Execution)
```

### Key Components

**Tokenizer Layer** (`modules/gdscript/gdscript_tokenizer.*`)
- Breaks source into tokens (keywords, identifiers, operators)
- **Critical files:**
  - `gdscript_tokenizer.h` - Token::Type enum
  - `gdscript_tokenizer.cpp` - KEYWORDS macro, token_names array
  - `gdscript_tokenizer_buffer.h` - TOKENIZER_VERSION

**Parser Layer** (`modules/gdscript/gdscript_parser.*`)
- Builds AST from tokens
- **Critical structures:**
  - Node base class with Type enum
  - Specific node types (ClassNode, FunctionNode, etc.)
  - DataType system (NOT Variant::Type!)
  - Parse rules array (MUST match Token enum exactly)

**Analyzer Layer** (`modules/gdscript/gdscript_analyzer.*`)
- Type checking and validation
- Resolves types, checks member access
- Generates warnings

**Type System** (Two parallel systems!)
1. **Variant::Type** - Core engine types (INT, FLOAT, OBJECT, etc.)
2. **DataType::Kind** - GDScript-specific types (CLASS, ENUM, STRUCT, etc.)

---

## Adding New Language Features

### The 7-Step Checklist

When adding a new keyword (like `struct`):

#### ✅ Step 1: Add Token Type
**File:** `modules/gdscript/gdscript_tokenizer.h`
```cpp
enum Type {
    SIGNAL,
    STATIC,
    STRUCT,    // ← Add alphabetically in keywords section
    SUPER,
    // ...
};
```

#### ✅ Step 2: Add Token Name
**File:** `modules/gdscript/gdscript_tokenizer.cpp`
```cpp
static const char *token_names[] = {
    "signal", // SIGNAL
    "static", // STATIC
    "struct", // STRUCT ← Add in EXACT same order as enum
    "super",  // SUPER
    // ...
};
```

**CRITICAL:** Order MUST match Token::Type enum! Static assert will fail otherwise.

#### ✅ Step 3: Register Keyword
**File:** `modules/gdscript/gdscript_tokenizer.cpp`
```cpp
#define KEYWORDS                              \
    KEYWORD("signal", Token::SIGNAL)          \
    KEYWORD("static", Token::STATIC)          \
    KEYWORD("struct", Token::STRUCT)          \ // ← Alphabetical by first letter
    KEYWORD("super", Token::SUPER)            \
```

#### ✅ Step 4: Bump Tokenizer Version
**File:** `modules/gdscript/gdscript_tokenizer_buffer.h`
```cpp
static constexpr uint32_t TOKENIZER_VERSION = 102; // Was 101
```

**Why:** Invalidates cached tokenizer data when Token enum changes.

#### ✅ Step 5: Add Parse Rule
**File:** `modules/gdscript/gdscript_parser.cpp` (around line 4360)
```cpp
static ParseRule rules[] = {
    // ...
    { nullptr, nullptr, PREC_NONE }, // SIGNAL
    { nullptr, nullptr, PREC_NONE }, // STATIC
    { nullptr, nullptr, PREC_NONE }, // STRUCT ← MUST match token order!
    { &GDScriptParser::parse_call, nullptr, PREC_NONE }, // SUPER
    // ...
};
```

**CRITICAL:** Array size MUST match Token::TK_MAX. Static assert enforces this.

#### ✅ Step 6: Add AST Node Type
**File:** `modules/gdscript/gdscript_parser.h`

Forward declaration:
```cpp
struct SignalNode;
struct StructNode;  // ← Add forward declaration
struct SubscriptNode;
```

Node::Type enum:
```cpp
enum Type {
    SIGNAL,
    STRUCT,    // ← Add alphabetically
    SUBSCRIPT,
    // ...
};
```

#### ✅ Step 7: Implement Node Class
**File:** `modules/gdscript/gdscript_parser.h` (after similar nodes)
```cpp
struct StructNode : public Node {
    IdentifierNode *identifier = nullptr;
    Vector<Member> members;
    HashMap<StringName, int> members_indices;
    
    StructNode() {
        type = STRUCT;  // ← MUST set type!
    }
};
```

---

## Parser Deep Dive

### Parse Rules Array

**Location:** `modules/gdscript/gdscript_parser.cpp:4288`

**Structure:**
```cpp
struct ParseRule {
    ParseFunction prefix;   // For prefix operators (e.g., -x, !x)
    ParseFunction infix;    // For binary/postfix (e.g., x+y, x[i])
    Precedence precedence;  // For infix
};
```

**Common Patterns:**
- Keywords: `{ nullptr, nullptr, PREC_NONE }`
- Literals: `{ &parse_literal, nullptr, PREC_NONE }`
- Binary ops: `{ nullptr, &parse_binary_operator, PREC_* }`
- Prefix ops: `{ &parse_unary_operator, nullptr, PREC_NONE }`

### ClassNode::Member Pattern

**Use case:** Adding new top-level declarations (like struct, enum, signal)

**File:** `modules/gdscript/gdscript_parser.h:561`

```cpp
struct Member {
    enum Type {
        CLASS,
        CONSTANT,
        FUNCTION,
        SIGNAL,
        VARIABLE,
        ENUM,
        STRUCT,  // ← Add your type
        GROUP,
    };
    
    Type type = UNDEFINED;
    
    union {
        ClassNode *m_class = nullptr;
        EnumNode *m_enum;
        StructNode *m_struct;  // ← Add pointer
        // ...
    };
    
    // ✅ MUST add constructor!
    Member(StructNode *p_struct) {
        type = STRUCT;
        m_struct = p_struct;
    }
    
    // ✅ MUST update all methods!
    String get_name() const {
        switch (type) {
            case STRUCT:
                return m_struct->identifier->name;
            // ...
        }
    }
    
    String get_type_name() const {
        switch (type) {
            case STRUCT:
                return "struct";
            // ...
        }
    }
    
    int get_line() const {
        switch (type) {
            case STRUCT:
                return m_struct->start_line;
            // ...
        }
    }
    
    DataType get_datatype() const {
        switch (type) {
            case STRUCT:
                return DataType(); // Or construct appropriate type
            // ...
        }
    }
    
    Node *get_source_node() const {
        switch (type) {
            case STRUCT:
                return m_struct;
            // ...
        }
    }
};
```

**Gotcha:** Forget any of these and you get compilation errors. Use compiler warnings as a checklist!

### parse_class_body Integration

**Location:** `modules/gdscript/gdscript_parser.cpp:1096`

```cpp
void GDScriptParser::parse_class_body(bool p_is_multiline) {
    while (!class_end && !is_at_end()) {
        switch (current.type) {
            case GDScriptTokenizer::Token::STRUCT:
                parse_class_member(&GDScriptParser::parse_struct, 
                                 AnnotationInfo::NONE, 
                                 "struct");
                break;
            // ...
        }
    }
}
```

### Implementing parse_* Functions

**Pattern to follow:** Look at similar features (signal, enum)

```cpp
GDScriptParser::StructNode *GDScriptParser::parse_struct(bool p_is_static) {
    StructNode *struct_node = alloc_node<StructNode>();
    
    // 1. Parse identifier
    if (!consume(Token::IDENTIFIER, R"(Expected name after "struct".)")) {
        complete_extents(struct_node);
        return nullptr;
    }
    struct_node->identifier = parse_identifier();
    
    // 2. Expect colon
    if (!consume(Token::COLON, R"(Expected ":" after struct name.)")) {
        complete_extents(struct_node);
        return nullptr;
    }
    
    // 3. Handle multiline blocks
    bool multiline = match(Token::NEWLINE);
    if (multiline && !consume(Token::INDENT, R"(Expected indented block.)")) {
        complete_extents(struct_node);
        return nullptr;
    }
    
    // 4. Parse body
    while (!check(Token::DEDENT) && !is_at_end()) {
        // Parse members...
    }
    
    // 5. Close block
    if (multiline) {
        consume(Token::DEDENT, R"(Missing unindent.)");
    }
    
    // 6. Always complete extents!
    complete_extents(struct_node);
    return struct_node;
}
```

**Key points:**
- Always call `complete_extents()` before returning
- Early returns should `return nullptr` after `complete_extents()`
- Use `R"(...)"` for error messages (raw strings avoid escaping)
- Check `!is_at_end()` to avoid infinite loops

---

## Type System

### Two Parallel Systems (IMPORTANT!)

**1. Variant::Type** - Engine Core Types
```cpp
enum Type {
    NIL, BOOL, INT, FLOAT, STRING,
    VECTOR2, VECTOR3, OBJECT,
    ARRAY, DICTIONARY,
    // ... (33 types total)
};
```

**Location:** `core/variant/variant.h`  
**Use for:** Runtime values, engine-level types  
**DON'T modify unless:** Absolutely necessary (affects entire engine)

**2. DataType::Kind** - GDScript Type System
```cpp
enum Kind {
    BUILTIN,   // Maps to Variant::Type
    NATIVE,    // Native C++ classes
    SCRIPT,    // Script types
    CLASS,     // GDScript classes
    ENUM,      // GDScript enums
    STRUCT,    // GDScript structs ← Add here!
    VARIANT,   // Any type
    RESOLVING,
    UNRESOLVED,
};
```

**Location:** `modules/gdscript/gdscript_parser.h:106`  
**Use for:** Compile-time type checking, annotations  
**Modify when:** Adding GDScript-specific types (enums, structs, etc.)

### DataType Structure

```cpp
struct DataType {
    Kind kind = UNRESOLVED;
    
    // Different fields used based on kind:
    Variant::Type builtin_type;  // For BUILTIN
    StringName native_type;      // For NATIVE
    StringName enum_type;        // For ENUM
    StringName struct_type;      // For STRUCT ← Add this
    ClassNode *class_type;       // For CLASS
    StructNode *struct_definition; // For STRUCT ← Add this
    Ref<Script> script_type;     // For SCRIPT
    
    String to_string() const;    // ← Update this!
    PropertyInfo to_property_info(const String &p_name) const;
};
```

### Updating DataType Methods

**Always update these when adding to Kind enum:**

```cpp
// 1. to_string()
String GDScriptParser::DataType::to_string() const {
    switch (kind) {
        case STRUCT:
            if (struct_definition && struct_definition->identifier) {
                return struct_definition->identifier->name;
            }
            return struct_type.operator String();
        // ...
    }
}

// 2. to_property_info() - if needed
PropertyInfo GDScriptParser::DataType::to_property_info(const String &p_name) const {
    // May need struct handling
}

// 3. Any comparison/equality methods
```

---

## Build System

### SCons Basics

```bash
# Clean build (slow, ~30-40 min)
python -m SCons platform=windows target=editor

# Dev build (faster compilation, larger binary, debug symbols)
python -m SCons platform=windows target=editor dev_build=yes

# Skip D3D12 (avoids dependency installation)
python -m SCons platform=windows target=editor dev_build=yes d3d12=no

# Parallel jobs (use CPU count - 1)
python -m SCons platform=windows target=editor --jobs=8

# Incremental build (only changed files, ~30 seconds to 5 minutes)
# Just run same command again after changes
```

### Build Output Location

```
godot/bin/godot.windows.editor.dev.x86_64.exe  (dev build)
godot/bin/godot.windows.editor.x86_64.exe      (release build)
```

### Common Build Issues

**Issue:** `static_assert failed: 'Amount of parse rules don't match'`  
**Cause:** Added token but didn't add parse rule  
**Fix:** Add entry to `rules[]` array in `gdscript_parser.cpp`

**Issue:** `error C2440: cannot convert from 'T *' to 'ClassNode::Member'`  
**Cause:** Added node type but didn't add Member constructor  
**Fix:** Add `Member(YourNode *p_node) { type = YOUR_TYPE; your_node = p_node; }`

**Issue:** `warning C4061: enumerator not explicitly handled`  
**Cause:** Added enum value but didn't update switch statements  
**Fix:** Search for all switches on that enum, add cases

**Issue:** Git index.lock file exists  
**Cause:** Previous git command interrupted  
**Fix:** `Remove-Item .git\index.lock -Force`

### Compilation Time Expectations

- **Full build:** 30-40 minutes (first time)
- **Incremental (1 file):** 30 seconds
- **Incremental (module):** 2-5 minutes
- **Incremental (core change):** 10-20 minutes

### Finding What to Rebuild

```bash
# Check what will be built
python -m SCons platform=windows target=editor --dry-run

# Module-specific rebuild
python -m SCons platform=windows target=editor --jobs=8 modules/gdscript

# Compile only without linking (faster for syntax checks)
python -m SCons platform=windows target=editor --jobs=8 --no-link
```

---

## Testing Patterns

### Test File Structure

**Location:** `modules/gdscript/tests/scripts/`

**Directories:**
- `parser/features/` - Valid syntax tests
- `parser/errors/` - Expected error tests
- `parser/warnings/` - Warning tests
- `analyzer/` - Type checking tests
- `runtime/` - Execution tests

### Test File Format

**Feature Test** (`features/struct_basic.gd`):
```gdscript
# Test description
struct Point:
    var x: int
    var y: int

func test():
    print("Test passed")
```

**Expected Output** (`features/struct_basic.out`):
```
GDTEST_OK
Test passed
```

**Error Test** (`errors/struct_invalid.gd`):
```gdscript
struct Invalid:
    func method():  # Error: methods not allowed
        pass

func test():
    pass
```

**Expected Output** (`errors/struct_invalid.out`):
```
GDTEST_PARSER_ERROR
Structs cannot have methods.
```

### Running Tests

```bash
# All tests
./bin/godot.windows.editor.dev.x86_64.exe --test

# Specific filter
./bin/godot.windows.editor.dev.x86_64.exe --test --test-filter="*gdscript*"
./bin/godot.windows.editor.dev.x86_64.exe --test --test-filter="*parser*"
./bin/godot.windows.editor.dev.x86_64.exe --test --test-filter="*struct*"

# Single test file
./bin/godot.windows.editor.dev.x86_64.exe --gdscript-test modules/gdscript/tests/scripts/parser/features/struct_basic.gd
```

### Test Naming Conventions

- `feature_name.gd` - Basic feature test
- `feature_name_variant.gd` - Specific variant
- `error_description.gd` - Error case
- `warning_description.gd` - Warning case

**Examples:**
- `struct_basic.gd`
- `struct_default_values.gd`
- `struct_methods_not_allowed.gd`
- `struct_keyword_reserved.gd`

---

## Common Gotchas

### 1. Parse Rules Array Sync

**Problem:** Added token but forgot parse rule
```cpp
// Added to Token::Type:
STRUCT,

// Forgot to add to rules[]:
static ParseRule rules[] = {
    // ... missing STRUCT entry
};
```

**Error:** `static_assert failed: 'Amount of parse rules'`  
**Solution:** Add `{ nullptr, nullptr, PREC_NONE }` for STRUCT

### 2. Token Names Array Order

**Problem:** token_names[] doesn't match Token::Type order
```cpp
enum Type {
    STATIC,
    STRUCT,  // Position 121
};

static const char *token_names[] = {
    "static",
    // Missing "struct"!
    "super",
};
```

**Error:** Runtime assertion or wrong token names  
**Solution:** ALWAYS keep arrays in sync, use static_assert

### 3. Member Constructor Missing

**Problem:** Added node to union but no constructor
```cpp
union {
    StructNode *m_struct;  // Added
};

// Missing:
Member(StructNode *p_struct) { ... }
```

**Error:** `error C2440: cannot convert`  
**Solution:** Add constructor for every union member type

### 4. Incomplete Switch Statements

**Problem:** Added enum value, didn't update switches
```cpp
enum Type { STRUCT };  // Added

String get_name() const {
    switch (type) {
        case ENUM:
            return m_enum->name;
        // Missing STRUCT case!
    }
}
```

**Error:** `warning C4061` or runtime crashes  
**Solution:** Search entire file for switches on that enum

### 5. Forgetting complete_extents()

**Problem:** Parser function doesn't set node extent
```cpp
StructNode *parse_struct() {
    StructNode *node = alloc_node<StructNode>();
    // ... parsing ...
    return node;  // ❌ Missing complete_extents()!
}
```

**Error:** Incorrect error reporting, wrong line numbers  
**Solution:** ALWAYS call `complete_extents()` before returning

### 6. CRLF vs LF Line Endings

**Problem:** Git warnings about CRLF replacement
```
warning: in the working copy of 'file.gd', CRLF will be replaced by LF
```

**Not an error** but clutters output.  
**Solution:** Use `.gitattributes` or configure editor for LF

### 7. Git Index Lock

**Problem:** `fatal: Unable to create '.git/index.lock': File exists`  
**Cause:** Previous git command interrupted  
**Solution:**
```powershell
Remove-Item .git\index.lock -Force
```

### 8. Variant vs DataType Confusion

**Problem:** Adding types to wrong system
```cpp
// ❌ Wrong: Adding to Variant::Type (affects entire engine)
enum Type {
    OBJECT,
    STRUCT,  // Don't do this!
};

// ✅ Right: Adding to DataType::Kind (GDScript only)
enum Kind {
    CLASS,
    STRUCT,  // Do this!
};
```

**Solution:** Use DataType::Kind for GDScript types, Variant::Type for engine types

---

## Code Patterns

### Pattern: Finding Existing Implementations

When adding a feature, find similar features:

```bash
# Find enum implementation (similar to struct)
grep -r "EnumNode" modules/gdscript/

# Find how signal is parsed
grep -r "parse_signal" modules/gdscript/

# Find all DataType::Kind switches
grep -r "switch.*kind" modules/gdscript/
```

### Pattern: Safe Node Allocation

```cpp
NodeType *node = alloc_node<NodeType>();  // Use this, not new!

if (!parse_something()) {
    complete_extents(node);  // Always complete
    return nullptr;          // Return null on error
}

complete_extents(node);      // Always complete before return
return node;
```

### Pattern: Error Messages

```cpp
// Use R"(...)" for messages with quotes
push_error(R"(Expected ":" after struct name.)");

// Use vformat for variable insertion
push_error(vformat(R"(Member "%s" already declared.)", name));

// Context-specific messages
consume(Token::COLON, R"(Expected ":" after struct name.)");
```

### Pattern: Member Lookup

```cpp
struct StructNode {
    Vector<Member> members;
    HashMap<StringName, int> members_indices;
    
    bool has_member(const StringName &p_name) const {
        return members_indices.has(p_name);
    }
    
    const Member &get_member(const StringName &p_name) const {
        return members[members_indices[p_name]];
    }
    
    void add_member(const Member &p_member) {
        members_indices[p_member.name] = members.size();
        members.push_back(p_member);
    }
};
```

**Why HashMap?** O(1) lookup vs O(n) linear search in Vector.

### Pattern: Duplicate Detection

```cpp
if (members_indices.has(member_name)) {
    push_error(vformat(R"(Member "%s" already declared.)", member_name));
} else {
    members_indices[member_name] = members.size();
    members.push_back(member);
}
```

### Pattern: Multiline Block Parsing

```cpp
bool multiline = match(Token::NEWLINE);

if (multiline && !consume(Token::INDENT, R"(Expected indent.)")) {
    complete_extents(node);
    return nullptr;
}

// Parse body...
while (!check(Token::DEDENT) && !is_at_end()) {
    // ...
}

if (multiline) {
    consume(Token::DEDENT, R"(Missing unindent.)");
}
```

---

## Quick Reference

### Files to Know

| File | Purpose |
|------|---------|
| `gdscript_tokenizer.h` | Token enum |
| `gdscript_tokenizer.cpp` | Keyword registration |
| `gdscript_tokenizer_buffer.h` | Version number |
| `gdscript_parser.h` | AST nodes, DataType |
| `gdscript_parser.cpp` | Parsing logic, parse rules |
| `gdscript_analyzer.h` | Type checking |
| `gdscript_analyzer.cpp` | Type resolution |
| `gdscript_compiler.cpp` | Bytecode generation |

### Enums to Update

When adding language feature:
- `Token::Type` - Add keyword token
- `Node::Type` - Add AST node type
- `ClassNode::Member::Type` - If top-level declaration
- `DataType::Kind` - If new type system concept

### Methods to Update

When adding to `DataType::Kind`:
- `DataType::to_string()`
- `DataType::to_property_info()` (maybe)
- Any comparison/hashing methods

When adding to `ClassNode::Member::Type`:
- `Member` constructor
- `get_name()`
- `get_type_name()`
- `get_line()`
- `get_datatype()`
- `get_source_node()`

### Build Commands Cheat Sheet

```bash
# Fresh build
python -m SCons platform=windows target=editor dev_build=yes d3d12=no --jobs=8

# Incremental rebuild
python -m SCons platform=windows target=editor dev_build=yes d3d12=no --jobs=8

# Run tests
./bin/godot.windows.editor.dev.x86_64.exe --test --test-filter="*struct*"

# Check what would build
python -m SCons --dry-run
```

---

## MCP Integration Ideas

This guide forms the knowledge base for a "Godot Expert" MCP that could:

**Code Navigation:**
```
"Find all switches on DataType::Kind"
"Show me how parse_signal works"
"Where is struct type resolution?"
```

**Code Generation:**
```
"Add a new token called 'trait' following Godot patterns"
"Generate a parser function for trait declarations"
"Create tests for trait syntax"
```

**Architecture Queries:**
```
"Explain the difference between Variant::Type and DataType::Kind"
"How does the analyzer resolve types?"
"What's the parser → analyzer → compiler flow?"
```

**Build Assistant:**
```
"Compile only the GDScript module"
"Run struct-related tests"
"What files need updating when I add an enum value?"
```

---

## Conclusion

Godot's architecture is well-structured but has strict synchronization requirements. The key to success:

1. **Follow existing patterns** - Don't invent new approaches
2. **Update ALL the things** - Enums require updating multiple places
3. **Use compiler as checklist** - Warnings tell you what's missing
4. **Test incrementally** - Compile often to catch errors early
5. **Document as you go** - Future you will thank you

**Remember:** When in doubt, grep for how similar features work!

---

**Version History:**
- 1.0 (2026-01-19): Initial version based on struct implementation experience

**Authors:** GitHub Copilot CLI  
**License:** Same as Godot Engine (MIT)
