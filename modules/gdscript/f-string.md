(Presumably f-strings would require documentation in the godot-docs repo eventually, but for
conversation purposes, I'm dropping a description of how they work here for now.)

# Formatted Strings

Formatted Strings (f-strings) are an implementation of string interpolation, borrowed somewhat from
the Python pattern.

## Functionality

A formatted string supports adding expressions to be printed directly inside the string between
braces. We call a brace-wrapped expression a *slot*. Those slots are evaluated as if they were live
code in the local context, so you can access variables, perform computation, make function calls,
etc. The string form of each such expression will be placed into the formatted string in that slot's
location.

Examples:

* `f"2 + 3 = {2+3}"` -> `"2 + 3 = 5"`
* `f"Interpolate a variable {PI}"` -> `"Interpolate a variable 3.14159265358979"`
* `f"Make a function call {"x".repeat(5)}"` -> `"Make a function call xxxxx"`

If you want a literal brace in your formatted string without creating a slot, you can escape it by
using a double-brace.  e.g.

* `f"will not evaluate {{this as a slot}}"` -> `"will not evaluate {this as a slot}"`.

We support a few alternate styles of formatted strings.

1. **Alternate quote characters**. You can use either double or single quotes to wrap the string.
   e.g. `f"foo"` or `f'foo'`. The close quote(s) must match the open ones.
2. **Raw strings**. You can use raw string processing to ignore escaping characters. e.g.
   `fr"foo\tbar"`.
3. **Multiline strings**. You can format multiline strings. e.g.:

   ```python
   f"""foo
   text{slot}
   bar"""
   ```

4. Any combination of these works as well. e.g. a raw multiline string using single quotes.

## Implementation

This implementation is really two changes in one.

1. There is a **rewrite of the Tokenizer** to change the invariants around where `_advance()` is
   called. As it was, many tokenizer methods looked backwards in the stream to understand the
   context of the currently processed token. With this rewrite, we ask that each tokenizing method
   only `_advance()` past characters that it will directly generate a Token for. Now any tokenizing
   method on entry can expect `_peek()` to reveal the first character for its token. If it will be
   calling other methods to generate other tokens, it should put/leave the stream in a state so they
   can start their processing at `_peek()` (and not e.g. `_peek(-1)`). This change was prompted
   because strings are now generated in multiple ways (regular strings and formatted strings), and
   we need to support sub-string processing (not including the delimiting quotes). It was much
   easier to think about the appropriate `_advance()` behavior of methods when we guarantee that
   they start looking at the current character. This change cascaded into changing the entire
   Tokenizer, but IMO it's much easier to read and understand that each method is correct when you
   can see the `_advance()` calls in the same function that returns the Token that includes those
   characters.

2. The **formatted string implementation** comes as changes to the Tokenizer, the Parser, and the Compiler.
   * The **Tokenizer** turns formatted strings into the following tokens (largely matching the
     [grammar proposed by
     Ivorforce](https://github.com/godotengine/godot-proposals/issues/157#issuecomment-2367779793)):

     ```text
        FORMATTED_STRING_BEGIN
        (STRING_LITERAL | (BRACE_OPEN EXPRESSION BRACE_CLOSE)) *
        FORMATTED_STRING_END
     ```

     This means that we need to generate several tokens for each formatted string, involving a call
     to scan() to produce each token. We do this by maintaining formatted-string-parsing state in
     the tokenizer, which we maintain in a couple of places.

     * `int fstring_parse_depth;`

     This keeps track of the current overall state of the formatted string tokenization. As we
     process a formatted string, we update the depth. The depth represents one of three main states:
     `NOT_IN_FORMATTED_STRING`, `NOT_IN_SLOT`, and `IN_SLOT`. These states correspond to the
     following places in this example code:

     ```text
         NOT_IN_FORMATTED_STRING        IN_SLOT        IN_SLOT    NOT_IN_FORMATTED_STRING
               |                          |               |                 |
         code code code f"text text {expression} text {expression} text" code code
                             |                     |                 |
                         NOT_IN_SLOT           NOT_IN_SLOT       NOT_IN_SLOT
     ```

     So as we're processing, typically we're in `NOT_IN_FORMATTED_STRING` mode. When we see the
     start of a formatted string, we change to `NOT_IN_SLOT` mode. When we see a slot, we move to
     `IN_SLOT`. When the slot ends we move back into `NOT_IN_SLOT`. etc.

     The `scan()` tokenizer uses this mode in the following ways. When we enter `scan()` in
     `NOT_IN_FORMATTED_STRING` mode, it processes tokens as it always has. When we enter `scan()` in
     `NOT_IN_SLOT` mode, we extract a string literal up to the next slot start or the end-of-string
     quote. We do this early in `scan()`, before the whitespace-skipping logic. When we enter
     `scan()` in `IN_SLOT` mode, we process code tokens as if it wasn't within a formatted string,
     since those tokens will later be composed into an expression. The one exception to this is that
     if we see a close-brace token, we emit that and return to `NOT_IN_SLOT` mode.

     Given that any code can be used in composing a slot expression, people may choose to put a
     formatted string inside there as well. Since each formatted string can separately choose to use
     any of the described variants (quote char, raw, multiline), we maintain a stack of these
     configurations, one for each nested formatted string:

     ```cpp
     struct FormattedStringConfig {
         bool is_raw;
         bool is_multiline;
         char32_t quote_char;
     }
     List<FormattedStringConfig> fstring_config_stack;
     ```

     *Paren matching*: one other bit of implementation that's worth noting is that since the
     tokenizer is given free reign to handle tokens within an expression, we want to make sure that
     the user doesn't start matching parens that were opened outside of the formatted string. So
     when we start processing a formatted string we add a new type of paren (`'<'`) to the paren
     stack to prevent parens from outside the formatted string being closed inside of a slot.

   * The **Parser** turns those tokens into a FormattedStringNode which consists of a list of
     template pieces (a text fragment or slot expression). Nested formatted strings are handled
     implicitly by the existing `parse_expression()` logic.

   * The **Compiler** composes a `String::format` function call from the template pieces in the
     FormattedStringNode. e.g.
       * `FormattedStringNode("foo {2+3} bar {6-2}")` -> `"foo {0} bar {1}".format([2+3, 6-2])`

## Open issues

* I don't understand how the parser extent mechanism works (`reset_extents`, `update_extents`,
  `complete_extents`). I just dropped them in until things seemed to work. Is there documentation
  for these somewhere?
* I haven't investigated error handling (compile time or run time).
* I haven't investigated syntax highlighting.
* There is not currently any support for formatting specifiers like f"{12345:,}" or f"{var=}". I
  propose that this support should be handled primarily by the String.format function. That is, we
  should first add formatting to String.format (e.g. `"foo {0:.2f}".format([1.2345]))`, and then we
  can forward the format spec from f-string slots to that function (e.g. `f"foo {1.2345:.2f}"`).
