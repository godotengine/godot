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

// langsvr implements a Language Server for the SPIRV assembly language.
package main

import (
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/KhronosGroup/SPIRV-Tools/utils/vscode/src/parser"
	"github.com/KhronosGroup/SPIRV-Tools/utils/vscode/src/schema"

	"github.com/KhronosGroup/SPIRV-Tools/utils/vscode/src/lsp/jsonrpc2"
	lsp "github.com/KhronosGroup/SPIRV-Tools/utils/vscode/src/lsp/protocol"
)

const (
	enableDebugLogging = false
)

// rSpy is a reader 'spy' that wraps an io.Reader, and logs all data that passes
// through it.
type rSpy struct {
	prefix string
	r      io.Reader
}

func (s rSpy) Read(p []byte) (n int, err error) {
	n, err = s.r.Read(p)
	log.Printf("%v %v", s.prefix, string(p[:n]))
	return n, err
}

// wSpy is a reader 'spy' that wraps an io.Writer, and logs all data that passes
// through it.
type wSpy struct {
	prefix string
	w      io.Writer
}

func (s wSpy) Write(p []byte) (n int, err error) {
	n, err = s.w.Write(p)
	log.Printf("%v %v", s.prefix, string(p))
	return n, err
}

// main entry point.
func main() {
	log.SetOutput(ioutil.Discard)
	if enableDebugLogging {
		// create a log file in the executable's directory.
		if logfile, err := os.Create(path.Join(path.Dir(os.Args[0]), "log.txt")); err == nil {
			defer logfile.Close()
			log.SetOutput(logfile)
		}
	}

	log.Println("language server started")

	stream := jsonrpc2.NewHeaderStream(rSpy{"IDE", os.Stdin}, wSpy{"LS", os.Stdout})
	s := server{
		files: map[string]*file{},
	}
	s.ctx, s.conn, s.client = lsp.NewServer(context.Background(), stream, &s)
	if err := s.conn.Run(s.ctx); err != nil {
		log.Panicln(err)
		os.Exit(1)
	}

	log.Println("language server stopped")
}

type server struct {
	ctx    context.Context
	conn   *jsonrpc2.Conn
	client lsp.Client

	files      map[string]*file
	filesMutex sync.Mutex
}

// file represents a source file
type file struct {
	fullRange parser.Range
	res       parser.Results
}

// tokAt returns the parser token at the given position lp
func (f *file) tokAt(lp lsp.Position) *parser.Token {
	toks := f.res.Tokens
	p := parser.Position{Line: int(lp.Line) + 1, Column: int(lp.Character) + 1}
	i := sort.Search(len(toks), func(i int) bool { return p.LessThan(toks[i].Range.End) })
	if i == len(toks) {
		return nil
	}
	if toks[i].Range.Contains(p) {
		return toks[i]
	}
	return nil
}

func (s *server) DidChangeWorkspaceFolders(ctx context.Context, p *lsp.DidChangeWorkspaceFoldersParams) error {
	log.Println("server.DidChangeWorkspaceFolders()")
	return nil
}
func (s *server) Initialized(ctx context.Context, p *lsp.InitializedParams) error {
	log.Println("server.Initialized()")
	return nil
}
func (s *server) Exit(ctx context.Context) error {
	log.Println("server.Exit()")
	return nil
}
func (s *server) DidChangeConfiguration(ctx context.Context, p *lsp.DidChangeConfigurationParams) error {
	log.Println("server.DidChangeConfiguration()")
	return nil
}
func (s *server) DidOpen(ctx context.Context, p *lsp.DidOpenTextDocumentParams) error {
	log.Println("server.DidOpen()")
	return s.processFile(ctx, p.TextDocument.URI, p.TextDocument.Text)
}
func (s *server) DidChange(ctx context.Context, p *lsp.DidChangeTextDocumentParams) error {
	log.Println("server.DidChange()")
	return s.processFile(ctx, p.TextDocument.URI, p.ContentChanges[0].Text)
}
func (s *server) DidClose(ctx context.Context, p *lsp.DidCloseTextDocumentParams) error {
	log.Println("server.DidClose()")
	return nil
}
func (s *server) DidSave(ctx context.Context, p *lsp.DidSaveTextDocumentParams) error {
	log.Println("server.DidSave()")
	return nil
}
func (s *server) WillSave(ctx context.Context, p *lsp.WillSaveTextDocumentParams) error {
	log.Println("server.WillSave()")
	return nil
}
func (s *server) DidChangeWatchedFiles(ctx context.Context, p *lsp.DidChangeWatchedFilesParams) error {
	log.Println("server.DidChangeWatchedFiles()")
	return nil
}
func (s *server) Progress(ctx context.Context, p *lsp.ProgressParams) error {
	log.Println("server.Progress()")
	return nil
}
func (s *server) SetTraceNotification(ctx context.Context, p *lsp.SetTraceParams) error {
	log.Println("server.SetTraceNotification()")
	return nil
}
func (s *server) LogTraceNotification(ctx context.Context, p *lsp.LogTraceParams) error {
	log.Println("server.LogTraceNotification()")
	return nil
}
func (s *server) Implementation(ctx context.Context, p *lsp.ImplementationParams) ([]lsp.Location, error) {
	log.Println("server.Implementation()")
	return nil, nil
}
func (s *server) TypeDefinition(ctx context.Context, p *lsp.TypeDefinitionParams) ([]lsp.Location, error) {
	log.Println("server.TypeDefinition()")
	return nil, nil
}
func (s *server) DocumentColor(ctx context.Context, p *lsp.DocumentColorParams) ([]lsp.ColorInformation, error) {
	log.Println("server.DocumentColor()")
	return nil, nil
}
func (s *server) ColorPresentation(ctx context.Context, p *lsp.ColorPresentationParams) ([]lsp.ColorPresentation, error) {
	log.Println("server.ColorPresentation()")
	return nil, nil
}
func (s *server) FoldingRange(ctx context.Context, p *lsp.FoldingRangeParams) ([]lsp.FoldingRange, error) {
	log.Println("server.FoldingRange()")
	return nil, nil
}
func (s *server) Declaration(ctx context.Context, p *lsp.DeclarationParams) ([]lsp.DeclarationLink, error) {
	log.Println("server.Declaration()")
	return nil, nil
}
func (s *server) SelectionRange(ctx context.Context, p *lsp.SelectionRangeParams) ([]lsp.SelectionRange, error) {
	log.Println("server.SelectionRange()")
	return nil, nil
}
func (s *server) Initialize(ctx context.Context, p *lsp.ParamInitia) (*lsp.InitializeResult, error) {
	log.Println("server.Initialize()")
	res := lsp.InitializeResult{
		Capabilities: lsp.ServerCapabilities{
			TextDocumentSync: lsp.TextDocumentSyncOptions{
				OpenClose: true,
				Change:    lsp.Full, // TODO: Implement incremental
			},
			HoverProvider:              true,
			DefinitionProvider:         true,
			ReferencesProvider:         true,
			RenameProvider:             true,
			DocumentFormattingProvider: true,
		},
	}
	return &res, nil
}
func (s *server) Shutdown(ctx context.Context) error {
	log.Println("server.Shutdown()")
	return nil
}
func (s *server) WillSaveWaitUntil(ctx context.Context, p *lsp.WillSaveTextDocumentParams) ([]lsp.TextEdit, error) {
	log.Println("server.WillSaveWaitUntil()")
	return nil, nil
}
func (s *server) Completion(ctx context.Context, p *lsp.CompletionParams) (*lsp.CompletionList, error) {
	log.Println("server.Completion()")
	return nil, nil
}
func (s *server) Resolve(ctx context.Context, p *lsp.CompletionItem) (*lsp.CompletionItem, error) {
	log.Println("server.Resolve()")
	return nil, nil
}
func (s *server) Hover(ctx context.Context, p *lsp.HoverParams) (*lsp.Hover, error) {
	log.Println("server.Hover()")
	f := s.getFile(p.TextDocument.URI)
	if f == nil {
		return nil, fmt.Errorf("Unknown file")
	}

	if tok := f.tokAt(p.Position); tok != nil {
		sb := strings.Builder{}
		switch v := f.res.Mappings[tok].(type) {
		default:
			sb.WriteString(fmt.Sprintf("<Unhandled type '%T'>", v))
		case *parser.Instruction:
			sb.WriteString(fmt.Sprintf("```\n%v\n```", v.Opcode.Opname))
		case *parser.Identifier:
			sb.WriteString(fmt.Sprintf("```\n%v\n```", v.Definition.Range.Text(f.res.Lines)))
		case *parser.Operand:
			if v.Name != "" {
				sb.WriteString(strings.Trim(v.Name, `'`))
				sb.WriteString("\n\n")
			}

			switch v.Kind.Category {
			case schema.OperandCategoryBitEnum:
			case schema.OperandCategoryValueEnum:
				sb.WriteString("```\n")
				sb.WriteString(strings.Trim(v.Kind.Kind, `'`))
				sb.WriteString("\n```")
			case schema.OperandCategoryID:
				if s := tok.Text(f.res.Lines); s != "" {
					if id, ok := f.res.Identifiers[s]; ok && id.Definition != nil {
						sb.WriteString("```\n")
						sb.WriteString(id.Definition.Range.Text(f.res.Lines))
						sb.WriteString("\n```")
					}
				}
			case schema.OperandCategoryLiteral:
			case schema.OperandCategoryComposite:
			}
		case nil:
		}

		if sb.Len() > 0 {
			res := lsp.Hover{
				Contents: lsp.MarkupContent{
					Kind:  "markdown",
					Value: sb.String(),
				},
			}
			return &res, nil
		}
	}

	return nil, nil
}
func (s *server) SignatureHelp(ctx context.Context, p *lsp.SignatureHelpParams) (*lsp.SignatureHelp, error) {
	log.Println("server.SignatureHelp()")
	return nil, nil
}
func (s *server) Definition(ctx context.Context, p *lsp.DefinitionParams) ([]lsp.Location, error) {
	log.Println("server.Definition()")
	if f := s.getFile(p.TextDocument.URI); f != nil {
		if tok := f.tokAt(p.Position); tok != nil {
			if s := tok.Text(f.res.Lines); s != "" {
				if id, ok := f.res.Identifiers[s]; ok {
					loc := lsp.Location{
						URI:   p.TextDocument.URI,
						Range: rangeToLSP(id.Definition.Range),
					}
					return []lsp.Location{loc}, nil
				}
			}
		}
	}
	return nil, nil
}
func (s *server) References(ctx context.Context, p *lsp.ReferenceParams) ([]lsp.Location, error) {
	log.Println("server.References()")
	if f := s.getFile(p.TextDocument.URI); f != nil {
		if tok := f.tokAt(p.Position); tok != nil {
			if s := tok.Text(f.res.Lines); s != "" {
				if id, ok := f.res.Identifiers[s]; ok {
					locs := make([]lsp.Location, len(id.References))
					for i, r := range id.References {
						locs[i] = lsp.Location{
							URI:   p.TextDocument.URI,
							Range: rangeToLSP(r.Range),
						}
					}
					return locs, nil
				}
			}
		}
	}
	return nil, nil
}
func (s *server) DocumentHighlight(ctx context.Context, p *lsp.DocumentHighlightParams) ([]lsp.DocumentHighlight, error) {
	log.Println("server.DocumentHighlight()")
	return nil, nil
}
func (s *server) DocumentSymbol(ctx context.Context, p *lsp.DocumentSymbolParams) ([]lsp.DocumentSymbol, error) {
	log.Println("server.DocumentSymbol()")
	return nil, nil
}
func (s *server) CodeAction(ctx context.Context, p *lsp.CodeActionParams) ([]lsp.CodeAction, error) {
	log.Println("server.CodeAction()")
	return nil, nil
}
func (s *server) Symbol(ctx context.Context, p *lsp.WorkspaceSymbolParams) ([]lsp.SymbolInformation, error) {
	log.Println("server.Symbol()")
	return nil, nil
}
func (s *server) CodeLens(ctx context.Context, p *lsp.CodeLensParams) ([]lsp.CodeLens, error) {
	log.Println("server.CodeLens()")
	return nil, nil
}
func (s *server) ResolveCodeLens(ctx context.Context, p *lsp.CodeLens) (*lsp.CodeLens, error) {
	log.Println("server.ResolveCodeLens()")
	return nil, nil
}
func (s *server) DocumentLink(ctx context.Context, p *lsp.DocumentLinkParams) ([]lsp.DocumentLink, error) {
	log.Println("server.DocumentLink()")
	return nil, nil
}
func (s *server) ResolveDocumentLink(ctx context.Context, p *lsp.DocumentLink) (*lsp.DocumentLink, error) {
	log.Println("server.ResolveDocumentLink()")
	return nil, nil
}
func (s *server) Formatting(ctx context.Context, p *lsp.DocumentFormattingParams) ([]lsp.TextEdit, error) {
	log.Println("server.Formatting()")
	if f := s.getFile(p.TextDocument.URI); f != nil {
		// Start by measuring the distance from the start of each line to the
		// first opcode on that line.
		lineInstOffsets, maxInstOffset, instOffset, curOffset := []int{}, 0, 0, -1
		for _, t := range f.res.Tokens {
			curOffset++ // whitespace between tokens
			switch t.Type {
			case parser.Ident:
				if _, isInst := schema.Opcodes[t.Text(f.res.Lines)]; isInst && instOffset == 0 {
					instOffset = curOffset
					continue
				}
			case parser.Newline:
				lineInstOffsets = append(lineInstOffsets, instOffset)
				if instOffset > maxInstOffset {
					maxInstOffset = instOffset
				}
				curOffset, instOffset = -1, 0
			default:
				curOffset += utf8.RuneCountInString(t.Text(f.res.Lines))
			}
		}
		lineInstOffsets = append(lineInstOffsets, instOffset)

		// Now rewrite each of the lines, adding padding at the start of the
		// line for alignment.
		sb, newline := strings.Builder{}, true
		for _, t := range f.res.Tokens {
			if newline {
				newline = false
				indent := maxInstOffset - lineInstOffsets[0]
				lineInstOffsets = lineInstOffsets[1:]
				switch t.Type {
				case parser.Newline, parser.Comment:
				default:
					for s := 0; s < indent; s++ {
						sb.WriteRune(' ')
					}
				}
			} else if t.Type != parser.Newline {
				sb.WriteString(" ")
			}

			sb.WriteString(t.Text(f.res.Lines))
			if t.Type == parser.Newline {
				newline = true
			}
		}

		formatted := sb.String()

		// Every good file ends with a single new line.
		formatted = strings.TrimRight(formatted, "\n") + "\n"

		return []lsp.TextEdit{
			{
				Range:   rangeToLSP(f.fullRange),
				NewText: formatted,
			},
		}, nil
	}
	return nil, nil
}
func (s *server) RangeFormatting(ctx context.Context, p *lsp.DocumentRangeFormattingParams) ([]lsp.TextEdit, error) {
	log.Println("server.RangeFormatting()")
	return nil, nil
}
func (s *server) OnTypeFormatting(ctx context.Context, p *lsp.DocumentOnTypeFormattingParams) ([]lsp.TextEdit, error) {
	log.Println("server.OnTypeFormatting()")
	return nil, nil
}
func (s *server) Rename(ctx context.Context, p *lsp.RenameParams) (*lsp.WorkspaceEdit, error) {
	log.Println("server.Rename()")
	if f := s.getFile(p.TextDocument.URI); f != nil {
		if tok := f.tokAt(p.Position); tok != nil {
			if s := tok.Text(f.res.Lines); s != "" {
				if id, ok := f.res.Identifiers[s]; ok {
					changes := make([]lsp.TextEdit, len(id.References))
					for i, r := range id.References {
						changes[i].Range = rangeToLSP(r.Range)
						changes[i].NewText = p.NewName
					}
					m := map[string][]lsp.TextEdit{}
					m[p.TextDocument.URI] = changes
					return &lsp.WorkspaceEdit{Changes: &m}, nil
				}
			}
		}
	}
	return nil, nil
}
func (s *server) PrepareRename(ctx context.Context, p *lsp.PrepareRenameParams) (*lsp.Range, error) {
	log.Println("server.PrepareRename()")
	return nil, nil
}
func (s *server) ExecuteCommand(ctx context.Context, p *lsp.ExecuteCommandParams) (interface{}, error) {
	log.Println("server.ExecuteCommand()")
	return nil, nil
}

func (s *server) processFile(ctx context.Context, uri, source string) error {
	log.Println("server.DidOpen()")
	res, err := parser.Parse(source)
	if err != nil {
		return err
	}
	fullRange := parser.Range{
		Start: parser.Position{Line: 1, Column: 1},
		End:   parser.Position{Line: len(res.Lines), Column: utf8.RuneCountInString(res.Lines[len(res.Lines)-1]) + 1},
	}

	s.filesMutex.Lock()
	s.files[uri] = &file{
		fullRange: fullRange,
		res:       res,
	}
	s.filesMutex.Unlock()

	dp := lsp.PublishDiagnosticsParams{URI: uri, Diagnostics: make([]lsp.Diagnostic, len(res.Diagnostics))}
	for i, d := range res.Diagnostics {
		dp.Diagnostics[i] = diagnosticToLSP(d)
	}
	s.client.PublishDiagnostics(ctx, &dp)
	return nil
}

func (s *server) getFile(uri string) *file {
	s.filesMutex.Lock()
	defer s.filesMutex.Unlock()
	return s.files[uri]
}

func diagnosticToLSP(d parser.Diagnostic) lsp.Diagnostic {
	return lsp.Diagnostic{
		Range:    rangeToLSP(d.Range),
		Severity: severityToLSP(d.Severity),
		Message:  d.Message,
	}
}

func severityToLSP(s parser.Severity) lsp.DiagnosticSeverity {
	switch s {
	case parser.SeverityError:
		return lsp.SeverityError
	case parser.SeverityWarning:
		return lsp.SeverityWarning
	case parser.SeverityInformation:
		return lsp.SeverityInformation
	case parser.SeverityHint:
		return lsp.SeverityHint
	default:
		log.Panicf("Invalid severity '%d'", int(s))
		return lsp.SeverityError
	}
}

func rangeToLSP(r parser.Range) lsp.Range {
	return lsp.Range{
		Start: positionToLSP(r.Start),
		End:   positionToLSP(r.End),
	}
}

func positionToLSP(r parser.Position) lsp.Position {
	return lsp.Position{
		Line:      float64(r.Line - 1),
		Character: float64(r.Column - 1),
	}
}
