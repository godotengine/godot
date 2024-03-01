// Copyright (C) 2019 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package parser implements a SPIR-V assembly parser.
package parser

import (
	"fmt"
	"io"
	"log"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/KhronosGroup/SPIRV-Tools/utils/vscode/src/schema"
)

// Type is an enumerator of token types.
type Type int

// Type enumerators
const (
	Ident  Type = iota // Foo
	PIdent             // %32, %foo
	Integer
	Float
	String
	Operator
	Comment
	Newline
)

func (t Type) String() string {
	switch t {
	case Ident:
		return "Ident"
	case PIdent:
		return "PIdent"
	case Integer:
		return "Integer"
	case Float:
		return "Float"
	case String:
		return "String"
	case Operator:
		return "Operator"
	case Comment:
		return "Comment"
	default:
		return "<unknown>"
	}
}

// Token represents a single lexed token.
type Token struct {
	Type  Type
	Range Range
}

func (t Token) String() string { return fmt.Sprintf("{%v %v}", t.Type, t.Range) }

// Text returns the tokens text from the source.
func (t Token) Text(lines []string) string { return t.Range.Text(lines) }

// Range represents an interval in a text file.
type Range struct {
	Start Position
	End   Position
}

func (r Range) String() string { return fmt.Sprintf("[%v %v]", r.Start, r.End) }

// Text returns the text for the given Range in the provided lines.
func (r Range) Text(lines []string) string {
	sl, sc := r.Start.Line-1, r.Start.Column-1
	if sl < 0 || sc < 0 || sl > len(lines) || sc > len(lines[sl]) {
		return fmt.Sprintf("<invalid start position %v>", r.Start)
	}
	el, ec := r.End.Line-1, r.End.Column-1
	if el < 0 || ec < 0 || el > len(lines) || ec > len(lines[sl]) {
		return fmt.Sprintf("<invalid end position %v>", r.End)
	}

	sb := strings.Builder{}
	if sl != el {
		sb.WriteString(lines[sl][sc:])
		for l := sl + 1; l < el; l++ {
			sb.WriteString(lines[l])
		}
		sb.WriteString(lines[el][:ec])
	} else {
		sb.WriteString(lines[sl][sc:ec])
	}
	return sb.String()
}

// Contains returns true if p is in r.
func (r Range) Contains(p Position) bool {
	return !(p.LessThan(r.Start) || p.GreaterThan(r.End))
}

func (r *Range) grow(o Range) {
	if !r.Start.IsValid() || o.Start.LessThan(r.Start) {
		r.Start = o.Start
	}
	if !r.End.IsValid() || o.End.GreaterThan(r.End) {
		r.End = o.End
	}
}

// Position holds a line and column position in a text file.
type Position struct {
	Line, Column int
}

func (p Position) String() string { return fmt.Sprintf("%v:%v", p.Line, p.Column) }

// IsValid returns true if the position has a line and column greater than 1.
func (p Position) IsValid() bool { return p.Line > 0 && p.Column > 0 }

// LessThan returns true iff o is before p.
func (p Position) LessThan(o Position) bool {
	switch {
	case !p.IsValid() || !o.IsValid():
		return false
	case p.Line < o.Line:
		return true
	case p.Line > o.Line:
		return false
	case p.Column < o.Column:
		return true
	default:
		return false
	}
}

// GreaterThan returns true iff o is greater than p.
func (p Position) GreaterThan(o Position) bool {
	switch {
	case !p.IsValid() || !o.IsValid():
		return false
	case p.Line > o.Line:
		return true
	case p.Line < o.Line:
		return false
	case p.Column > o.Column:
		return true
	default:
		return false
	}
}

type lexer struct {
	source string
	lexerState
	diags []Diagnostic
	e     error
}

type lexerState struct {
	offset int      // byte offset in source
	toks   []*Token // all the lexed tokens
	pos    Position // current position
}

// err appends an fmt.Printf style error into l.diags for the given token.
func (l *lexer) err(tok *Token, msg string, args ...interface{}) {
	rng := Range{}
	if tok != nil {
		rng = tok.Range
	}
	l.diags = append(l.diags, Diagnostic{
		Range:    rng,
		Severity: SeverityError,
		Message:  fmt.Sprintf(msg, args...),
	})
}

// next returns the next rune, or io.EOF if the last rune has already been
// consumed.
func (l *lexer) next() rune {
	if l.offset >= len(l.source) {
		l.e = io.EOF
		return 0
	}
	r, n := utf8.DecodeRuneInString(l.source[l.offset:])
	l.offset += n
	if n == 0 {
		l.e = io.EOF
		return 0
	}
	if r == '\n' {
		l.pos.Line++
		l.pos.Column = 1
	} else {
		l.pos.Column++
	}
	return r
}

// save returns the current lexerState.
func (l *lexer) save() lexerState {
	return l.lexerState
}

// restore restores the current lexer state with s.
func (l *lexer) restore(s lexerState) {
	l.lexerState = s
}

// pident processes the PIdent token at the current position.
// The lexer *must* know the next token is a PIdent before calling.
func (l *lexer) pident() {
	tok := &Token{Type: PIdent, Range: Range{Start: l.pos, End: l.pos}}
	if r := l.next(); r != '%' {
		log.Fatalf("lexer expected '%%', got '%v'", r)
		return
	}
	for l.e == nil {
		s := l.save()
		r := l.next()
		if !isAlphaNumeric(r) && r != '_' {
			l.restore(s)
			break
		}
	}
	tok.Range.End = l.pos
	l.toks = append(l.toks, tok)
}

// numberOrIdent processes the Ident, Float or Integer token at the current
// position.
func (l *lexer) numberOrIdent() {
	const Unknown Type = -1
	tok := &Token{Type: Unknown, Range: Range{Start: l.pos, End: l.pos}}
loop:
	for l.e == nil {
		s := l.save()
		r := l.next()
		switch {
		case r == '-', r == '+', isNumeric(r):
			continue
		case isAlpha(r), r == '_':
			switch tok.Type {
			case Unknown:
				tok.Type = Ident
			case Float, Integer:
				l.err(tok, "invalid number")
				return
			}
		case r == '.':
			switch tok.Type {
			case Unknown:
				tok.Type = Float
			default:
				l.restore(s)
				break loop
			}
		default:
			if tok.Type == Unknown {
				tok.Type = Integer
			}
			l.restore(s)
			break loop
		}
	}
	tok.Range.End = l.pos
	l.toks = append(l.toks, tok)
}

// string processes the String token at the current position.
// The lexer *must* know the next token is a String before calling.
func (l *lexer) string() {
	tok := &Token{Type: String, Range: Range{Start: l.pos, End: l.pos}}
	if r := l.next(); r != '"' {
		log.Fatalf("lexer expected '\"', got '%v'", r)
		return
	}
	escape := false
	for l.e == nil {
		switch l.next() {
		case '"':
			if !escape {
				tok.Range.End = l.pos
				l.toks = append(l.toks, tok)
				return
			}
		case '\\':
			escape = !escape
		default:
			escape = false
		}
	}
}

// operator processes the Operator token at the current position.
// The lexer *must* know the next token is a Operator before calling.
func (l *lexer) operator() {
	tok := &Token{Type: Operator, Range: Range{Start: l.pos, End: l.pos}}
	for l.e == nil {
		switch l.next() {
		case '=', '|':
			tok.Range.End = l.pos
			l.toks = append(l.toks, tok)
			return
		}
	}
}

// lineComment processes the Comment token at the current position.
// The lexer *must* know the next token is a Comment before calling.
func (l *lexer) lineComment() {
	tok := &Token{Type: Comment, Range: Range{Start: l.pos, End: l.pos}}
	if r := l.next(); r != ';' {
		log.Fatalf("lexer expected ';', got '%v'", r)
		return
	}
	for l.e == nil {
		s := l.save()
		switch l.next() {
		case '\n':
			l.restore(s)
			tok.Range.End = l.pos
			l.toks = append(l.toks, tok)
			return
		}
	}
}

// newline processes the Newline token at the current position.
// The lexer *must* know the next token is a Newline before calling.
func (l *lexer) newline() {
	tok := &Token{Type: Newline, Range: Range{Start: l.pos, End: l.pos}}
	if r := l.next(); r != '\n' {
		log.Fatalf("lexer expected '\n', got '%v'", r)
		return
	}
	tok.Range.End = l.pos
	l.toks = append(l.toks, tok)
}

// lex returns all the tokens and diagnostics after lexing source.
func lex(source string) ([]*Token, []Diagnostic, error) {
	l := lexer{source: source, lexerState: lexerState{pos: Position{1, 1}}}

	lastPos := Position{}
	for l.e == nil {
		// Integrity check that the parser is making progress
		if l.pos == lastPos {
			log.Panicf("Parsing stuck at %v", l.pos)
		}
		lastPos = l.pos

		s := l.save()
		r := l.next()
		switch {
		case r == '%':
			l.restore(s)
			l.pident()
		case r == '+' || r == '-' || r == '_' || isAlphaNumeric(r):
			l.restore(s)
			l.numberOrIdent()
		case r == '"':
			l.restore(s)
			l.string()
		case r == '=', r == '|':
			l.restore(s)
			l.operator()
		case r == ';':
			l.restore(s)
			l.lineComment()
		case r == '\n':
			l.restore(s)
			l.newline()
		}
	}
	if l.e != nil && l.e != io.EOF {
		return nil, nil, l.e
	}
	return l.toks, l.diags, nil
}

func isNumeric(r rune) bool      { return unicode.IsDigit(r) }
func isAlpha(r rune) bool        { return unicode.IsLetter(r) }
func isAlphaNumeric(r rune) bool { return isAlpha(r) || isNumeric(r) }

type parser struct {
	lines          []string                    // all source lines
	toks           []*Token                    // all tokens
	diags          []Diagnostic                // parser emitted diagnostics
	idents         map[string]*Identifier      // identifiers by name
	mappings       map[*Token]interface{}      // tokens to semantic map
	extInstImports map[string]schema.OpcodeMap // extension imports by identifier
	insts          []*Instruction              // all instructions
}

func (p *parser) parse() error {
	for i := 0; i < len(p.toks); {
		if p.newline(i) || p.comment(i) {
			i++
			continue
		}
		if n := p.instruction(i); n > 0 {
			i += n
		} else {
			p.unexpected(i)
			i++
		}
	}
	return nil
}

// instruction parses the instruction starting at the i'th token.
func (p *parser) instruction(i int) (n int) {
	inst := &Instruction{}

	switch {
	case p.opcode(i) != nil:
		inst.Opcode = p.opcode(i)
		inst.Tokens = []*Token{p.tok(i)}
		p.mappings[p.tok(i)] = inst
		n++
	case p.opcode(i+2) != nil: // try '%id' '='
		inst.Result, inst.Opcode = p.pident(i), p.opcode(i+2)
		if inst.Result == nil || p.operator(i+1) != "=" {
			return 0
		}
		n += 3
		inst.Tokens = []*Token{p.tok(i), p.tok(i + 1), p.tok(i + 2)}
		p.mappings[p.tok(i+2)] = inst
	default:
		return
	}

	expectsResult := len(inst.Opcode.Operands) > 0 && IsResult(inst.Opcode.Operands[0].Kind)
	operands := inst.Opcode.Operands
	switch {
	case inst.Result != nil && !expectsResult:
		p.err(inst.Result, "'%s' does not have a result", inst.Opcode.Opname)
		return
	case inst.Result == nil && expectsResult:
		p.err(p.tok(i), "'%s' expects a result", inst.Opcode.Opname)
		return
	case inst.Result != nil && expectsResult:
		// Check the result is of the correct type
		o := inst.Opcode.Operands[0]
		p.operand(o.Name, o.Kind, i, false)
		operands = operands[1:]
		p.addIdentDef(inst.Result.Text(p.lines), inst, p.tok(i))
	}

	processOperand := func(o schema.Operand) bool {
		if p.newline(i + n) {
			return false
		}

		switch o.Quantifier {
		case schema.Once:
			if op, c := p.operand(o.Name, o.Kind, i+n, false); op != nil {
				inst.Tokens = append(inst.Tokens, op.Tokens...)
				n += c
			}
		case schema.ZeroOrOnce:
			if op, c := p.operand(o.Name, o.Kind, i+n, true); op != nil {
				inst.Tokens = append(inst.Tokens, op.Tokens...)
				n += c
			}
		case schema.ZeroOrMany:
			for !p.newline(i + n) {
				if op, c := p.operand(o.Name, o.Kind, i+n, true); op != nil {
					inst.Tokens = append(inst.Tokens, op.Tokens...)
					n += c
				} else {
					return false
				}
			}
		}
		return true
	}

	for _, o := range operands {
		if !processOperand(o) {
			break
		}

		if inst.Opcode == schema.OpExtInst && n == 4 {
			extImportTok, extNameTok := p.tok(i+n), p.tok(i+n+1)
			extImport := extImportTok.Text(p.lines)
			if extOpcodes, ok := p.extInstImports[extImport]; ok {
				extName := extNameTok.Text(p.lines)
				if extOpcode, ok := extOpcodes[extName]; ok {
					n += 2 // skip ext import, ext name
					for _, o := range extOpcode.Operands {
						if !processOperand(o) {
							break
						}
					}
				} else {
					p.err(extNameTok, "Unknown extension opcode '%s'", extName)
				}
			} else {
				p.err(extImportTok, "Expected identifier to OpExtInstImport")
			}
		}
	}

	for _, t := range inst.Tokens {
		inst.Range.grow(t.Range)
	}

	p.insts = append(p.insts, inst)

	if inst.Opcode == schema.OpExtInstImport && len(inst.Tokens) >= 4 {
		// Instruction is a OpExtInstImport. Keep track of this.
		extTok := inst.Tokens[3]
		extName := strings.Trim(extTok.Text(p.lines), `"`)
		extOpcodes, ok := schema.ExtOpcodes[extName]
		if !ok {
			p.err(extTok, "Unknown extension '%s'", extName)
		}
		extImport := inst.Result.Text(p.lines)
		p.extInstImports[extImport] = extOpcodes
	}

	return
}

// operand parses the operand with the name n, kind k, starting at the i'th
// token.
func (p *parser) operand(n string, k *schema.OperandKind, i int, optional bool) (*Operand, int) {
	tok := p.tok(i)
	if tok == nil {
		return nil, 0
	}

	op := &Operand{
		Name:   n,
		Kind:   k,
		Tokens: []*Token{tok},
	}
	p.mappings[tok] = op

	switch k.Category {
	case schema.OperandCategoryBitEnum, schema.OperandCategoryValueEnum:
		s := tok.Text(p.lines)
		for _, e := range k.Enumerants {
			if e.Enumerant == s {
				count := 1
				for _, param := range e.Parameters {
					p, c := p.operand(param.Name, param.Kind, i+count, false)
					if p != nil {
						op.Tokens = append(op.Tokens, p.Tokens...)
						op.Parameters = append(op.Parameters, p)
					}
					count += c
				}

				// Handle bitfield '|' chains
				if p.tok(i+count).Text(p.lines) == "|" {
					count++ // '|'
					p, c := p.operand(n, k, i+count, false)
					if p != nil {
						op.Tokens = append(op.Tokens, p.Tokens...)
						op.Parameters = append(op.Parameters, p)
					}
					count += c
				}

				return op, count
			}
		}
		if !optional {
			p.err(p.tok(i), "invalid operand value '%s'", s)
		}

		return nil, 0

	case schema.OperandCategoryID:
		id := p.pident(i)
		if id != nil {
			p.addIdentRef(p.tok(i))
			return op, 1
		}
		if !optional {
			p.err(p.tok(i), "operand requires id, got '%s'", tok.Text(p.lines))
		}
		return nil, 0

	case schema.OperandCategoryLiteral:
		switch tok.Type {
		case String, Integer, Float, Ident:
			return op, 1
		}
		if !optional {
			p.err(p.tok(i), "operand requires literal, got '%s'", tok.Text(p.lines))
		}
		return nil, 0

	case schema.OperandCategoryComposite:
		n := 1
		for _, b := range k.Bases {
			o, c := p.operand(b.Kind, b, i+n, optional)
			if o != nil {
				op.Tokens = append(op.Tokens, o.Tokens...)
			}
			n += c
		}
		return op, n

	default:
		p.err(p.tok(i), "OperandKind '%s' has unexpected category '%s'", k.Kind, k.Category)
		return nil, 0
	}
}

// tok returns the i'th token, or nil if i is out of bounds.
func (p *parser) tok(i int) *Token {
	if i < 0 || i >= len(p.toks) {
		return nil
	}
	return p.toks[i]
}

// opcode returns the schema.Opcode for the i'th token, or nil if the i'th token
// does not represent an opcode.
func (p *parser) opcode(i int) *schema.Opcode {
	if tok := p.ident(i); tok != nil {
		name := tok.Text(p.lines)
		if inst, found := schema.Opcodes[name]; found {
			return inst
		}
	}
	return nil
}

// operator returns the operator for the i'th token, or and empty string if the
// i'th token is not an operator.
func (p *parser) operator(i int) string {
	if tok := p.tok(i); tok != nil && tok.Type == Operator {
		return tok.Text(p.lines)
	}
	return ""
}

// ident returns the i'th token if it is an Ident, otherwise nil.
func (p *parser) ident(i int) *Token {
	if tok := p.tok(i); tok != nil && tok.Type == Ident {
		return tok
	}
	return nil
}

// pident returns the i'th token if it is an PIdent, otherwise nil.
func (p *parser) pident(i int) *Token {
	if tok := p.tok(i); tok != nil && tok.Type == PIdent {
		return tok
	}
	return nil
}

// comment returns true if the i'th token is a Comment, otherwise false.
func (p *parser) comment(i int) bool {
	if tok := p.tok(i); tok != nil && tok.Type == Comment {
		return true
	}
	return false
}

// newline returns true if the i'th token is a Newline, otherwise false.
func (p *parser) newline(i int) bool {
	if tok := p.tok(i); tok != nil && tok.Type == Newline {
		return true
	}
	return false
}

// unexpected emits an 'unexpected token error' for the i'th token.
func (p *parser) unexpected(i int) {
	p.err(p.toks[i], "syntax error: unexpected '%s'", p.toks[i].Text(p.lines))
}

// addIdentDef records the token definition for the instruction inst with the
// given id.
func (p *parser) addIdentDef(id string, inst *Instruction, def *Token) {
	i, existing := p.idents[id]
	if !existing {
		i = &Identifier{}
		p.idents[id] = i
	}
	if i.Definition == nil {
		i.Definition = inst
	} else {
		p.err(def, "id '%v' redeclared", id)
	}
}

// addIdentRef adds a identifier reference for the token ref.
func (p *parser) addIdentRef(ref *Token) {
	id := ref.Text(p.lines)
	i, existing := p.idents[id]
	if !existing {
		i = &Identifier{}
		p.idents[id] = i
	}
	i.References = append(i.References, ref)
}

// err appends an fmt.Printf style error into l.diags for the given token.
func (p *parser) err(tok *Token, msg string, args ...interface{}) {
	rng := Range{}
	if tok != nil {
		rng = tok.Range
	}
	p.diags = append(p.diags, Diagnostic{
		Range:    rng,
		Severity: SeverityError,
		Message:  fmt.Sprintf(msg, args...),
	})
}

// Parse parses the SPIR-V assembly string source, returning the parse results.
func Parse(source string) (Results, error) {
	toks, diags, err := lex(source)
	if err != nil {
		return Results{}, err
	}
	lines := strings.SplitAfter(source, "\n")
	p := parser{
		lines:          lines,
		toks:           toks,
		idents:         map[string]*Identifier{},
		mappings:       map[*Token]interface{}{},
		extInstImports: map[string]schema.OpcodeMap{},
	}
	if err := p.parse(); err != nil {
		return Results{}, err
	}
	diags = append(diags, p.diags...)
	return Results{
		Lines:       lines,
		Tokens:      toks,
		Diagnostics: p.diags,
		Identifiers: p.idents,
		Mappings:    p.mappings,
	}, nil
}

// IsResult returns true if k is used to store the result of an instruction.
func IsResult(k *schema.OperandKind) bool {
	switch k {
	case schema.OperandKindIdResult, schema.OperandKindIdResultType:
		return true
	default:
		return false
	}
}

// Results holds the output of Parse().
type Results struct {
	Lines       []string
	Tokens      []*Token
	Diagnostics []Diagnostic
	Identifiers map[string]*Identifier // identifiers by name
	Mappings    map[*Token]interface{} // tokens to semantic map
}

// Instruction describes a single instruction instance
type Instruction struct {
	Tokens   []*Token       // all the tokens that make up the instruction
	Result   *Token         // the token that represents the result of the instruction, or nil
	Operands []*Operand     // the operands of the instruction
	Range    Range          // the textual range of the instruction
	Opcode   *schema.Opcode // the opcode for the instruction
}

// Operand describes a single operand instance
type Operand struct {
	Name       string              // name of the operand
	Kind       *schema.OperandKind // kind of the operand
	Tokens     []*Token            // all the tokens that make up the operand
	Parameters []*Operand          // all the parameters for the operand
}

// Identifier describes a single, unique SPIR-V identifier (i.e. %32)
type Identifier struct {
	Definition *Instruction // where the identifier was defined
	References []*Token     // all the places the identifier was referenced
}

// Severity is an enumerator of diagnostic severities
type Severity int

// Severity levels
const (
	SeverityError Severity = iota
	SeverityWarning
	SeverityInformation
	SeverityHint
)

// Diagnostic holds a single diagnostic message that was generated while
// parsing.
type Diagnostic struct {
	Range    Range
	Severity Severity
	Message  string
}
