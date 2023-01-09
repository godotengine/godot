/* _nir_foreach_dest() needs to be ALWAYS_INLINE so that it can inline the
 * callback if it was declared with ALWAYS_INLINE.
 */
static ALWAYS_INLINE bool
_nir_foreach_dest(nir_instr *instr, nir_foreach_dest_cb cb, void *state)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return cb(&nir_instr_as_alu(instr)->dest.dest, state);
   case nir_instr_type_deref:
      return cb(&nir_instr_as_deref(instr)->dest, state);
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (nir_intrinsic_infos[intrin->intrinsic].has_dest)
         return cb(&intrin->dest, state);
      return true;
   }
   case nir_instr_type_tex:
      return cb(&nir_instr_as_tex(instr)->dest, state);
   case nir_instr_type_phi:
      return cb(&nir_instr_as_phi(instr)->dest, state);
   case nir_instr_type_parallel_copy: {
      nir_foreach_parallel_copy_entry(entry, nir_instr_as_parallel_copy(instr)) {
         if (!cb(&entry->dest, state))
            return false;
      }
      return true;
   }

   case nir_instr_type_load_const:
   case nir_instr_type_ssa_undef:
   case nir_instr_type_call:
   case nir_instr_type_jump:
      break;

   default:
      unreachable("Invalid instruction type");
      break;
   }

   return true;
}

static ALWAYS_INLINE bool
_nir_visit_src(nir_src *src, nir_foreach_src_cb cb, void *state)
{
   if (!cb(src, state))
      return false;
   if (!src->is_ssa && src->reg.indirect)
      return cb(src->reg.indirect, state);
   return true;
}

typedef struct {
   void *state;
   nir_foreach_src_cb cb;
} _nir_visit_dest_indirect_state;

static ALWAYS_INLINE bool
_nir_visit_dest_indirect(nir_dest *dest, void *_state)
{
   _nir_visit_dest_indirect_state *state = (_nir_visit_dest_indirect_state *) _state;

   if (!dest->is_ssa && dest->reg.indirect)
      return state->cb(dest->reg.indirect, state->state);

   return true;
}

static inline bool
nir_foreach_dest(nir_instr *instr, nir_foreach_dest_cb cb, void *state)
{
   return _nir_foreach_dest(instr, cb, state);
}

static inline bool
nir_foreach_src(nir_instr *instr, nir_foreach_src_cb cb, void *state)
{
   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++)
         if (!_nir_visit_src(&alu->src[i].src, cb, state))
            return false;
      break;
   }
   case nir_instr_type_deref: {
      nir_deref_instr *deref = nir_instr_as_deref(instr);

      if (deref->deref_type != nir_deref_type_var) {
         if (!_nir_visit_src(&deref->parent, cb, state))
            return false;
      }

      if (deref->deref_type == nir_deref_type_array ||
          deref->deref_type == nir_deref_type_ptr_as_array) {
         if (!_nir_visit_src(&deref->arr.index, cb, state))
            return false;
      }
      break;
   }
   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      unsigned num_srcs = nir_intrinsic_infos[intrin->intrinsic].num_srcs;
      for (unsigned i = 0; i < num_srcs; i++) {
         if (!_nir_visit_src(&intrin->src[i], cb, state))
            return false;
      }
      break;
   }
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      for (unsigned i = 0; i < tex->num_srcs; i++) {
         if (!_nir_visit_src(&tex->src[i].src, cb, state))
            return false;
      }
      break;
   }
   case nir_instr_type_call: {
      nir_call_instr *call = nir_instr_as_call(instr);
      for (unsigned i = 0; i < call->num_params; i++) {
         if (!_nir_visit_src(&call->params[i], cb, state))
            return false;
      }
      break;
   }
   case nir_instr_type_phi: {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_foreach_phi_src(src, phi) {
         if (!_nir_visit_src(&src->src, cb, state))
            return false;
      }
      break;
   }
   case nir_instr_type_parallel_copy: {
      nir_parallel_copy_instr *pc = nir_instr_as_parallel_copy(instr);
      nir_foreach_parallel_copy_entry(entry, pc) {
         if (!_nir_visit_src(&entry->src, cb, state))
            return false;
      }
      break;
   }
   case nir_instr_type_jump: {
      nir_jump_instr *jump = nir_instr_as_jump(instr);

      if (jump->type == nir_jump_goto_if && !_nir_visit_src(&jump->condition, cb, state))
         return false;
      return true;
   }

   case nir_instr_type_load_const:
   case nir_instr_type_ssa_undef:
      return true;

   default:
      unreachable("Invalid instruction type");
      break;
   }

   _nir_visit_dest_indirect_state dest_state;
   dest_state.state = state;
   dest_state.cb = cb;
   return _nir_foreach_dest(instr, _nir_visit_dest_indirect, &dest_state);
}
