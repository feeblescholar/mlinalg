#ifndef AST_ADD_OP_IPP
#define AST_ADD_OP_IPP

// Definitions for add_op.hpp. Do not include this file directly.

namespace mlinalg {

template <Expression L, Expression R>
    requires detail::CwiseCompatible<L, R>
constexpr auto AddOp<L, R>::coeff(Index row, Index col) const -> expr_scalar_t<L> {
    return this->lhs().coeff(row, col) + this->rhs().coeff(row, col);
}

} // namespace mlinalg

#endif // AST_ADD_OP_IPP
