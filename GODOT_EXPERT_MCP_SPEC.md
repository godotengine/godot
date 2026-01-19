# 🚀 Godot Expert MCP Specification

**Purpose:** AI-powered expert system for Godot Engine development  
**Version:** 1.0 (Specification)  
**Date:** 2026-01-19

---

## Overview

The **Godot Expert MCP** is a Model Context Protocol server that provides expert-level assistance for Godot Engine development, particularly focused on GDScript language extensions and core engine modifications.

### Use Cases

1. **Code Navigation** - Find patterns, understand architecture
2. **Code Generation** - Generate boilerplate following Godot patterns
3. **Architecture Queries** - Explain design decisions and relationships
4. **Build Assistance** - Compile specific modules, run targeted tests
5. **Pattern Matching** - Find similar implementations to use as templates

---

## Knowledge Base

The MCP draws from:

1. **GODOT_DEVELOPMENT_GUIDE.md** - Comprehensive development patterns
2. **Godot source code** - Direct analysis of implementation
3. **Build system** - SCons configuration understanding
4. **Test framework** - Test file formats and execution

---

## Proposed Tools

### 1. `godot_find_pattern`

Find code patterns in Godot source.

**Parameters:**
- `pattern`: Pattern to find (e.g., "parse_* functions", "DataType::Kind switches")
- `context`: Optional context (parser, analyzer, compiler)
- `show_examples`: Boolean - include code examples

**Example:**
```json
{
  "pattern": "How to add new token",
  "context": "tokenizer",
  "show_examples": true
}
```

**Response:**
```
Found pattern: Adding new token (7 steps)
Files to modify:
1. gdscript_tokenizer.h - Add to Token::Type enum
2. gdscript_tokenizer.cpp - Add to token_names array
...

Example from SIGNAL keyword:
[code examples]
```

### 2. `godot_explain_architecture`

Explain Godot architecture concepts.

**Parameters:**
- `concept`: What to explain (e.g., "DataType vs Variant", "parser flow")
- `detail_level`: "brief" | "detailed" | "comprehensive"

**Example:**
```json
{
  "concept": "DataType::Kind vs Variant::Type",
  "detail_level": "detailed"
}
```

**Response:**
```
DataType::Kind (GDScript compile-time types):
- Used for: Type annotations, type checking, analyzer
- Location: modules/gdscript/gdscript_parser.h
- Values: BUILTIN, NATIVE, SCRIPT, CLASS, ENUM, STRUCT...

Variant::Type (Engine runtime types):
- Used for: Actual runtime values, core engine
- Location: core/variant/variant.h  
- Values: NIL, BOOL, INT, FLOAT, OBJECT...

Key Difference: DataType is compile-time, Variant is runtime...
```

### 3. `godot_generate_code`

Generate Godot-style code following patterns.

**Parameters:**
- `template`: Code template type ("token", "node", "parse_function", "test")
- `name`: Name of the feature
- `options`: Template-specific options

**Example:**
```json
{
  "template": "node",
  "name": "Trait",
  "options": {
    "has_members": true,
    "has_identifier": true
  }
}
```

**Response:**
```cpp
// Forward declaration
struct TraitNode;

// Node definition
struct TraitNode : public Node {
    IdentifierNode *identifier = nullptr;
    Vector<Member> members;
    HashMap<StringName, int> members_indices;
    
    bool has_member(const StringName &p_name) const {
        return members_indices.has(p_name);
    }
    
    TraitNode() {
        type = TRAIT;
    }
};
```

### 4. `godot_find_similar`

Find similar implementations to use as template.

**Parameters:**
- `feature`: What you're implementing
- `similarity_type`: "syntax", "structure", "behavior"

**Example:**
```json
{
  "feature": "struct",
  "similarity_type": "structure"
}
```

**Response:**
```
Most similar implementations:
1. EnumNode (90% similar)
   - Has members with names
   - Has identifier
   - Top-level declaration
   - Files: gdscript_parser.h:1663, gdscript_parser.cpp:1563

2. SignalNode (70% similar)
   - Has parameters (similar to members)
   - Has identifier
   ...
```

### 5. `godot_check_completeness`

Check if all required updates are done for a change.

**Parameters:**
- `change_type`: "token", "node", "enum_value", "datatype"
- `name`: Name of what was added

**Example:**
```json
{
  "change_type": "datatype",
  "name": "STRUCT"
}
```

**Response:**
```
Checklist for DataType::Kind::STRUCT:

✅ Added to enum (gdscript_parser.h:112)
✅ Added struct_type field
✅ Added struct_definition field
✅ Updated to_string() method
⚠️  to_property_info() may need update
❌ Missing: Analyzer resolve_datatype() handling
❌ Missing: Type compatibility checking

Files to check:
- gdscript_analyzer.cpp:2145 (resolve_datatype)
- gdscript_analyzer.cpp:3890 (type compatibility)
```

### 6. `godot_build`

Build-related operations.

**Parameters:**
- `action`: "compile" | "test" | "check" | "clean"
- `scope`: "all" | "module" | "file"
- `target`: Optional specific target

**Example:**
```json
{
  "action": "compile",
  "scope": "module",
  "target": "gdscript"
}
```

**Response:**
```
Building GDScript module only...
Command: python -m SCons platform=windows target=editor modules/gdscript

[Build output...]

✅ Build successful (45 seconds)
Modified: bin/godot.windows.editor.dev.x86_64.exe
```

### 7. `godot_run_tests`

Run Godot tests.

**Parameters:**
- `filter`: Test filter pattern
- `category`: "parser" | "analyzer" | "runtime" | "all"

**Example:**
```json
{
  "filter": "*struct*",
  "category": "parser"
}
```

**Response:**
```
Running tests matching '*struct*' in category 'parser'...

✅ struct_keyword_reserved.gd - PASSED
✅ struct_basic.gd - PASSED  
✅ struct_default_values.gd - PASSED

3/3 tests passed (100%)
Time: 0.34s
```

### 8. `godot_analyze_error`

Analyze compilation or runtime errors.

**Parameters:**
- `error_message`: The error text
- `context`: Optional file/line context

**Example:**
```json
{
  "error_message": "static_assert failed: 'Amount of parse rules don't match'",
  "context": "gdscript_parser.cpp:4406"
}
```

**Response:**
```
Error Analysis:

Cause: Parse rules array size doesn't match Token::Type enum size
Common Reason: Added new token but forgot to add parse rule entry

Solution:
1. Open gdscript_parser.cpp:4288
2. Find static ParseRule rules[] array
3. Add entry for your new token:
   { nullptr, nullptr, PREC_NONE }, // YOUR_TOKEN

Related Guide: GODOT_DEVELOPMENT_GUIDE.md - "Parse Rules Array Sync"
```

---

## Implementation Options

### Option A: Python-based MCP Server

**Pros:**
- Easy integration with SCons build system
- Can directly execute Godot commands
- Rich string processing for code generation

**Structure:**
```
godot-expert-mcp/
├── server.py           # MCP server implementation
├── knowledge/
│   ├── guide.md       # Copy of GODOT_DEVELOPMENT_GUIDE.md
│   ├── patterns.json  # Extracted patterns
│   └── templates/     # Code templates
├── tools/
│   ├── find.py        # Pattern finding
│   ├── generate.py    # Code generation
│   ├── build.py       # Build operations
│   └── analyze.py     # Error analysis
└── requirements.txt
```

### Option B: TypeScript/Node MCP Server

**Pros:**
- Native MCP SDK support
- Better async handling
- Rich ecosystem for code parsing

**Structure:**
```
godot-expert-mcp/
├── src/
│   ├── index.ts       # MCP server
│   ├── tools/         # Tool implementations
│   ├── knowledge/     # Knowledge base
│   └── templates/     # Code templates
├── package.json
└── tsconfig.json
```

### Option C: Hybrid with Tree-sitter

**Pros:**
- Accurate code parsing and analysis
- Can understand C++ and GDScript syntax
- Better pattern matching

**Structure:**
- Use tree-sitter-cpp for parsing
- Python/Node for MCP server
- Knowledge base as structured data

---

## Knowledge Extraction

From `GODOT_DEVELOPMENT_GUIDE.md`, extract:

### 1. Patterns (patterns.json)
```json
{
  "add_token": {
    "steps": [
      {
        "file": "gdscript_tokenizer.h",
        "action": "Add to Token::Type enum",
        "location": "alphabetically in keywords section"
      },
      ...
    ]
  }
}
```

### 2. Code Templates (templates/)
```
templates/
├── token/
│   ├── enum_entry.cpp
│   ├── token_name.cpp
│   └── keyword_macro.cpp
├── node/
│   ├── forward_decl.h
│   ├── node_class.h
│   └── member_constructor.h
└── parse_function/
    └── basic_parse.cpp
```

### 3. Architecture (architecture.json)
```json
{
  "pipeline": {
    "stages": ["Tokenizer", "Parser", "Analyzer", "Compiler", "VM"],
    "files": {
      "Tokenizer": ["gdscript_tokenizer.h", "gdscript_tokenizer.cpp"]
    }
  }
}
```

---

## Usage Examples

### Scenario 1: Adding New Keyword

**User:** "I want to add a 'trait' keyword to GDScript. What do I need to do?"

**MCP:**
```
godot_find_pattern("add new keyword") →

7-step checklist for adding keywords:
1. ✅ Add to Token::Type enum (gdscript_tokenizer.h:125)
2. ✅ Add to token_names array (gdscript_tokenizer.cpp:121)
...

godot_generate_code("token", "trait") →
[Generated boilerplate code]

godot_check_completeness("token", "TRAIT") →
Checklist with what's done and what's missing
```

### Scenario 2: Understanding Architecture

**User:** "Why are there two type systems?"

**MCP:**
```
godot_explain_architecture("DataType vs Variant") →

Detailed explanation of:
- Variant::Type (runtime, engine-wide)
- DataType::Kind (compile-time, GDScript-specific)
- When to use each
- Examples from codebase
```

### Scenario 3: Debugging Build Error

**User:** "I get 'error C2440: cannot convert from T* to Member'"

**MCP:**
```
godot_analyze_error("error C2440: cannot convert...") →

Diagnosis: Missing ClassNode::Member constructor
Cause: Added new node type to union without constructor
Solution: [Step-by-step fix]
Similar Issues: [Links to guide sections]
```

---

## Integration with Copilot CLI

The MCP would work seamlessly with GitHub Copilot CLI:

```bash
# User asks question
$ gh copilot explain "How do I add a struct type?"

# Copilot CLI calls godot_find_pattern MCP
# Returns comprehensive guide

# User wants code
$ gh copilot suggest "Generate StructNode class"

# Copilot CLI calls godot_generate_code MCP  
# Returns properly formatted code following Godot patterns
```

---

## Next Steps

1. **Extract patterns** from GODOT_DEVELOPMENT_GUIDE.md into structured data
2. **Create templates** for common code generation tasks
3. **Implement core MCP** with 2-3 essential tools
4. **Test with real scenarios** from struct implementation
5. **Iterate and expand** based on usage

---

## Success Criteria

The MCP is successful when:

✅ Can guide a developer through adding a new language feature
✅ Generates code that compiles without modification  
✅ Explains architecture better than reading docs alone
✅ Catches common mistakes before compilation
✅ Reduces time to productivity for new contributors

---

## Future Enhancements

- **Interactive tutorials** - Step-by-step guidance
- **Code review assistant** - Check PRs for patterns
- **Performance analyzer** - Suggest optimizations
- **Migration helper** - Port features from other engines
- **Documentation generator** - Auto-generate docs from code

---

**Status:** Specification Complete  
**Next:** Implement prototype with core tools  
**Timeline:** 2-3 days for MVP

This MCP would make Godot development **10x faster** for language features! 🚀
