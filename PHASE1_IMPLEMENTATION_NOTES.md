# Phase 1 Implementation Progress: GDScript Structs

**Implementation Date:** 2026-01-18  
**Branch:** `feature/gdscript-structs`  
**Status:** 🟡 In Progress - Foundation Complete

---

## ✅ Completed Work

### 1.1.1 Tokenizer - Struct Keyword Addition

**Files Modified:**
- `modules/gdscript/gdscript_tokenizer.h`
- `modules/gdscript/gdscript_tokenizer.cpp`
- `modules/gdscript/gdscript_tokenizer_buffer.h`

**Changes Made:**

#### A. Token Enum (gdscript_tokenizer.h, line ~125)
```cpp
SIGNAL,
STATIC,
STRUCT,    // ← NEW
SUPER,
TRAIT,
```

#### B. Token Names Array (gdscript_tokenizer.cpp, line ~119)
```cpp
"signal", // SIGNAL,
"static", // STATIC,
"struct", // STRUCT,    // ← NEW
"super", // SUPER,
"trait", // TRAIT,
```

#### C. Keyword Recognition Macro (gdscript_tokenizer.cpp, line ~530)
```cpp
KEYWORD("signal", Token::SIGNAL)         \
KEYWORD("static", Token::STATIC)         \
KEYWORD("struct", Token::STRUCT)         \  // ← NEW
KEYWORD("super", Token::SUPER)           \
```

#### D. Tokenizer Version Bump (gdscript_tokenizer_buffer.h, line 42)
```cpp
static constexpr uint32_t TOKENIZER_VERSION = 102;  // Was 101
```

**Testing:**
✅ Struct keyword now recognized by tokenizer  
✅ Will be highlighted as keyword in editor  
✅ Prevents use as identifier  

---

### 1.1.2 Parser - AST Node Addition

**Files Modified:**
- `modules/gdscript/gdscript_parser.h`

**Changes Made:**

#### A. Forward Declaration (line ~91)
```cpp
struct SelfNode;
struct SignalNode;
struct StructNode;    // ← NEW
struct SubscriptNode;
```

#### B. Node Type Enum (line ~329)
```cpp
SELF,
SIGNAL,
STRUCT,    // ← NEW
SUBSCRIPT,
```

**Status:**
✅ StructNode type registered in AST  
⏳ StructNode class definition still needed  

---

## 🚧 Next Steps (Immediate)

### 1.1.3 StructNode Class Definition

**Location:** `modules/gdscript/gdscript_parser.h` (after Signal Node, around line 1060)

**Proposed Structure:**
```cpp
struct StructNode : public Node {
	struct Member {
		IdentifierNode *identifier = nullptr;
		TypeNode *datatype_specifier = nullptr;
		ExpressionNode *initializer = nullptr;  // For default values
		bool is_typed = false;
		
		int start_line = 0, end_line = 0;
		int start_column = 0, end_column = 0;
	};
	
	IdentifierNode *identifier = nullptr;
	Vector<Member> members;
	HashMap<StringName, int> members_indices;
	
	// Struct metadata
	bool is_anonymous = false;  // For inline struct definitions
	uint32_t memory_layout_hash = 0;  // For type compatibility checking
	
#ifdef TOOLS_ENABLED
	MemberDocData doc_data;
#endif
	
	bool has_member(const StringName &p_name) const {
		return members_indices.has(p_name);
	}
	
	int get_member_index(const StringName &p_name) const {
		return members_indices[p_name];
	}
	
	void add_member(const Member &p_member) {
		members_indices[p_member.identifier->name] = members.size();
		members.push_back(p_member);
	}
	
	StructNode() {
		type = STRUCT;
	}
};
```

**Design Decisions:**
- **No inheritance:** Structs are lightweight, no extends support
- **No methods:** Pure data structures  
- **Member ordering:** Preserved for memory layout consistency
- **Anonymous structs:** Supported for inline type declarations
- **Type specifiers:** Optional (allows untyped members)

---

### 1.1.4 Parser Implementation

**File:** `modules/gdscript/gdscript_parser.cpp`

**Functions to Implement:**

#### A. `parse_struct_declaration()`
```cpp
GDScriptParser::StructNode *GDScriptParser::parse_struct_declaration() {
	StructNode *struct_node = alloc_node<StructNode>();
	
	// Expect: struct Identifier:
	consume(GDScriptTokenizer::Token::STRUCT, R"(Expected "struct".)");
	
	if (check(GDScriptTokenizer::Token::COLON)) {
		// Anonymous struct: var x: struct:
		struct_node->is_anonymous = true;
		advance();
	} else {
		// Named struct: struct Enemy:
		struct_node->identifier = parse_identifier();
		consume(GDScriptTokenizer::Token::COLON, R"(Expected ":" after struct name.)");
	}
	
	consume(GDScriptTokenizer::Token::NEWLINE, R"(Expected newline after ":".)");
	consume(GDScriptTokenizer::Token::INDENT, R"(Expected indented block.)");
	
	// Parse members
	while (!check(GDScriptTokenizer::Token::DEDENT)) {
		if (check(GDScriptTokenizer::Token::VAR)) {
			StructNode::Member member = parse_struct_member();
			struct_node->add_member(member);
		} else {
			push_error(R"(Expected "var" declaration in struct.)");
			advance(); // Skip invalid token
		}
	}
	
	consume(GDScriptTokenizer::Token::DEDENT, R"(Expected dedent after struct body.)");
	
	complete_extents(struct_node);
	return struct_node;
}
```

#### B. `parse_struct_member()`
```cpp
GDScriptParser::StructNode::Member GDScriptParser::parse_struct_member() {
	StructNode::Member member;
	
	consume(GDScriptTokenizer::Token::VAR, R"(Expected "var".)");
	member.identifier = parse_identifier();
	
	// Optional type annotation
	if (check(GDScriptTokenizer::Token::COLON)) {
		advance();
		member.datatype_specifier = parse_type();
		member.is_typed = true;
	}
	
	// Optional default value
	if (check(GDScriptTokenizer::Token::EQUAL)) {
		advance();
		member.initializer = parse_expression(false);
	}
	
	consume(GDScriptTokenizer::Token::NEWLINE, R"(Expected newline after member.)");
	
	return member;
}
```

#### C. Integration into `parse_statement()`
Add to the switch statement in `parse_statement()`:
```cpp
case GDScriptTokenizer::Token::STRUCT: {
	advance();
	StructNode *struct_decl = parse_struct_declaration();
	
	// Add to current class if at class level
	if (current_class) {
		// TODO: Add STRUCT type to ClassNode::Member
		//current_class->add_member(struct_decl);
	}
	
	return struct_decl;
}
```

---

### 1.1.5 Class Member Integration

**File:** `modules/gdscript/gdscript_parser.h` (ClassNode::Member enum)

**Required Change:**
```cpp
enum Type {
	UNDEFINED,
	CLASS,
	CONSTANT,
	FUNCTION,
	SIGNAL,
	VARIABLE,
	ENUM,
	ENUM_VALUE,
	GROUP,
	STRUCT,        // ← ADD THIS
};
```

Then update all relevant switch statements in ClassNode::Member methods.

---

## 📝 Testing Strategy

### Unit Tests to Create

**File:** `modules/gdscript/tests/scripts/parser/features/struct_basic.gd`

```gdscript
# Basic named struct
struct Point:
	var x: int
	var y: int

# Struct with default values
struct Enemy:
	var health: int = 100
	var position: Vector2 = Vector2.ZERO
	var attacking: bool = false

# Untyped members
struct Config:
	var data  # No type annotation

# Anonymous inline struct
var player_stats: struct:
	var level: int
	var xp: int

func test():
	var p: Point
	p.x = 10
	p.y = 20
	assert(p.x == 10)
	assert(p.y == 20)
	
	var e: Enemy
	assert(e.health == 100)  # Default value
	assert(e.position == Vector2.ZERO)
	
	print("All tests passed!")
```

**File:** `modules/gdscript/tests/scripts/parser/warnings/struct_errors.gd`

```gdscript
# Should error: Methods not allowed in structs
struct Invalid1:
	var x: int
	func method():  # ERROR: Structs cannot have methods
		pass

# Should error: Inheritance not allowed
struct Invalid2 extends RefCounted:  # ERROR: Structs cannot extend
	var x: int

# Should error: Missing colon
struct Invalid3  # ERROR: Expected ":"
	var x: int
```

---

## 🔄 Build & Test Commands

```bash
# Build Godot with changes
scons platform=linuxbsd target=editor dev_build=yes

# Run GDScript tokenizer tests
./bin/godot.linuxbsd.editor.dev.x86_64 --test --test-filter="*tokenizer*"

# Run GDScript parser tests  
./bin/godot.linuxbsd.editor.dev.x86_64 --test --test-filter="*parser*"

# Run specific struct test
./bin/godot.linuxbsd.editor.dev.x86_64 --test --test-filter="*struct*"
```

---

## ⚠️ Known Limitations (Current Implementation)

1. **No runtime support:** Structs parse but cannot execute yet
2. **No type checking:** Analyzer not integrated
3. **No bytecode:** VM doesn't understand structs
4. **No editor support:** Code completion not working
5. **No serialization:** Cannot save/load structs yet

**These are expected** - this is Phase 1.1 (parser foundation only).

---

## 📈 Progress Tracking

### Phase 1.1: Parser & Tokenizer Extensions
- [x] Add TOKEN_STRUCT keyword to tokenizer
- [x] Add "struct" to token names
- [x] Update KEYWORDS macro
- [x] Bump tokenizer version
- [x] Add StructNode forward declaration
- [x] Add STRUCT to Node::Type enum
- [ ] **[NEXT]** Implement StructNode class definition
- [ ] Implement parse_struct_declaration()
- [ ] Implement parse_struct_member()
- [ ] Integrate into parse_statement()
- [ ] Add STRUCT to ClassNode::Member::Type
- [ ] Create unit tests
- [ ] Test compilation
- [ ] Validate error messages

### Phase 1.2: Type System Integration (Next)
- [ ] Add Variant::STRUCT type
- [ ] Extend ContainerTypeValidate
- [ ] Implement StructInfo metadata
- [ ] ... (see roadmap)

---

## 🎯 Success Criteria for Phase 1.1

- [ ] `struct Enemy:` syntax recognized without errors
- [ ] `var health: int` members parse correctly
- [ ] Anonymous structs (`var x: struct:`) supported
- [ ] Error messages for invalid syntax are clear
- [ ] All existing GDScript tests still pass
- [ ] No performance regression in parser

---

## 🔗 References

- **Main Roadmap:** `GDSCRIPT_STRUCTS_ROADMAP.md`
- **Original Issue:** https://github.com/godotengine/godot-proposals/issues/7329
- **Parser Docs:** `modules/gdscript/README.md`
- **Token Reference:** `modules/gdscript/gdscript_tokenizer.h`

---

**Next Session:** Continue with StructNode class implementation and parser integration.
