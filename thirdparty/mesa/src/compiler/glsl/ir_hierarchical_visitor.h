/* -*- c++ -*- */
/*
 * Copyright © 2010 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef IR_HIERARCHICAL_VISITOR_H
#define IR_HIERARCHICAL_VISITOR_H

/**
 * Enumeration values returned by visit methods to guide processing
 */
enum ir_visitor_status {
   visit_continue,		/**< Continue visiting as normal. */
   visit_continue_with_parent,	/**< Don't visit siblings, continue w/parent. */
   visit_stop			/**< Stop visiting immediately. */
};


#ifdef __cplusplus
/**
 * Base class of hierarchical visitors of IR instruction trees
 *
 * Hierarchical visitors differ from traditional visitors in a couple of
 * important ways.  Rather than having a single \c visit method for each
 * subclass in the composite, there are three kinds of visit methods.
 * Leaf-node classes have a traditional \c visit method.  Internal-node
 * classes have a \c visit_enter method, which is invoked just before
 * processing child nodes, and a \c visit_leave method which is invoked just
 * after processing child nodes.
 *
 * In addition, each visit method and the \c accept methods in the composite
 * have a return value which guides the navigation.  Any of the visit methods
 * can choose to continue visiting the tree as normal (by returning \c
 * visit_continue), terminate visiting any further nodes immediately (by
 * returning \c visit_stop), or stop visiting sibling nodes (by returning \c
 * visit_continue_with_parent).
 *
 * These two changes combine to allow navigation of children to be implemented
 * in the composite's \c accept method.  The \c accept method for a leaf-node
 * class will simply call the \c visit method, as usual, and pass its return
 * value on.  The \c accept method for internal-node classes will call the \c
 * visit_enter method, call the \c accept method of each child node, and,
 * finally, call the \c visit_leave method.  If any of these return a value
 * other that \c visit_continue, the correct action must be taken.
 *
 * The final benefit is that the hierarchical visitor base class need not be
 * abstract.  Default implementations of every \c visit, \c visit_enter, and
 * \c visit_leave method can be provided.  By default each of these methods
 * simply returns \c visit_continue.  This allows a significant reduction in
 * derived class code.
 *
 * For more information about hierarchical visitors, see:
 *
 *    http://c2.com/cgi/wiki?HierarchicalVisitorPattern
 *    http://c2.com/cgi/wiki?HierarchicalVisitorDiscussion
 */

class ir_hierarchical_visitor {
public:
   ir_hierarchical_visitor();

   /**
    * \name Visit methods for leaf-node classes
    */
   /*@{*/
   virtual ir_visitor_status visit(class ir_rvalue *);
   virtual ir_visitor_status visit(class ir_variable *);
   virtual ir_visitor_status visit(class ir_constant *);
   virtual ir_visitor_status visit(class ir_loop_jump *);
   virtual ir_visitor_status visit(class ir_barrier *);

   /**
    * ir_dereference_variable isn't technically a leaf, but it is treated as a
    * leaf here for a couple reasons.  By not automatically visiting the one
    * child ir_variable node from the ir_dereference_variable, ir_variable
    * nodes can always be handled as variable declarations.  Code that used
    * non-hierarchical visitors had to set an "in a dereference" flag to
    * determine how to handle an ir_variable.  By forcing the visitor to
    * handle the ir_variable within the ir_dereference_variable visitor, this
    * kludge can be avoided.
    *
    * In addition, I can envision no use for having separate enter and leave
    * methods.  Anything that could be done in the enter and leave methods
    * that couldn't just be done in the visit method.
    */
   virtual ir_visitor_status visit(class ir_dereference_variable *);
   /*@}*/

   /**
    * \name Visit methods for internal-node classes
    */
   /*@{*/
   virtual ir_visitor_status visit_enter(class ir_loop *);
   virtual ir_visitor_status visit_leave(class ir_loop *);
   virtual ir_visitor_status visit_enter(class ir_function_signature *);
   virtual ir_visitor_status visit_leave(class ir_function_signature *);
   virtual ir_visitor_status visit_enter(class ir_function *);
   virtual ir_visitor_status visit_leave(class ir_function *);
   virtual ir_visitor_status visit_enter(class ir_expression *);
   virtual ir_visitor_status visit_leave(class ir_expression *);
   virtual ir_visitor_status visit_enter(class ir_texture *);
   virtual ir_visitor_status visit_leave(class ir_texture *);
   virtual ir_visitor_status visit_enter(class ir_swizzle *);
   virtual ir_visitor_status visit_leave(class ir_swizzle *);
   virtual ir_visitor_status visit_enter(class ir_dereference_array *);
   virtual ir_visitor_status visit_leave(class ir_dereference_array *);
   virtual ir_visitor_status visit_enter(class ir_dereference_record *);
   virtual ir_visitor_status visit_leave(class ir_dereference_record *);
   virtual ir_visitor_status visit_enter(class ir_assignment *);
   virtual ir_visitor_status visit_leave(class ir_assignment *);
   virtual ir_visitor_status visit_enter(class ir_call *);
   virtual ir_visitor_status visit_leave(class ir_call *);
   virtual ir_visitor_status visit_enter(class ir_return *);
   virtual ir_visitor_status visit_leave(class ir_return *);
   virtual ir_visitor_status visit_enter(class ir_discard *);
   virtual ir_visitor_status visit_leave(class ir_discard *);
   virtual ir_visitor_status visit_enter(class ir_demote *);
   virtual ir_visitor_status visit_leave(class ir_demote *);
   virtual ir_visitor_status visit_enter(class ir_if *);
   virtual ir_visitor_status visit_leave(class ir_if *);
   virtual ir_visitor_status visit_enter(class ir_emit_vertex *);
   virtual ir_visitor_status visit_leave(class ir_emit_vertex *);
   virtual ir_visitor_status visit_enter(class ir_end_primitive *);
   virtual ir_visitor_status visit_leave(class ir_end_primitive *);
   /*@}*/


   /**
    * Utility function to process a linked list of instructions with a visitor
    */
   void run(struct exec_list *instructions);

   /**
    * Utility function to call both the leave and enter callback functions.
    * This is used for leaf nodes.
    */
   void call_enter_leave_callbacks(class ir_instruction *ir);

   /* Some visitors may need to insert new variable declarations and
    * assignments for portions of a subtree, which means they need a
    * pointer to the current instruction in the stream, not just their
    * node in the tree rooted at that instruction.
    *
    * This is implemented by visit_list_elements -- if the visitor is
    * not called by it, nothing good will happen.
    */
   class ir_instruction *base_ir;

   /**
    * Callback function that is invoked on entry to each node visited.
    *
    * \warning
    * Visitor classes derived from \c ir_hierarchical_visitor \b may \b not
    * invoke this function.  This can be used, for example, to cause the
    * callback to be invoked on every node type except one.
    */
   void (*callback_enter)(class ir_instruction *ir, void *data);

   /**
    * Callback function that is invoked on exit of each node visited.
    *
    * \warning
    * Visitor classes derived from \c ir_hierarchical_visitor \b may \b not
    * invoke this function.  This can be used, for example, to cause the
    * callback to be invoked on every node type except one.
    */
   void (*callback_leave)(class ir_instruction *ir, void *data);

   /**
    * Extra data parameter passed to the per-node callback_enter function
    */
   void *data_enter;

   /**
    * Extra data parameter passed to the per-node callback_leave function
    */
   void *data_leave;

   /**
    * Currently in the LHS of an assignment?
    *
    * This is set and cleared by the \c ir_assignment::accept method.
    */
   bool in_assignee;
};

void visit_tree(ir_instruction *ir,
		void (*callback_enter)(class ir_instruction *ir, void *data),
		void *data_enter,
		void (*callback_leave)(class ir_instruction *ir, void *data) = NULL,
		void *data_leave = NULL);

ir_visitor_status visit_list_elements(ir_hierarchical_visitor *v, exec_list *l,
                                      bool statement_list = true);
#endif /* __cplusplus */

#endif /* IR_HIERARCHICAL_VISITOR_H */
