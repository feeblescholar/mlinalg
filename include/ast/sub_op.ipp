#ifndef AST_SUB_OP_IPP
#define AST_SUB_OP_IPP

// Definitions for sub_op.hpp. Do not include this file directly.

namespace mlinalg {

template <Expression L, Expression R>
    requires detail::CwiseCompatible<L, R>
constexpr auto SubOp<L, R>::coeff(Index row, Index col) const -> expr_scalar_t<L> {
    return this->lhs().coeff(row, col) - this->rhs().coeff(row, col);
}

} // namespace mlinalg

#endif // AST_SUB_OP_IPP
