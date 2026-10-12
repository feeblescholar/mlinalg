#ifndef AST_TRANSPOSE_OP_IPP
#define AST_TRANSPOSE_OP_IPP

// Definitions for transpose_op.hpp. Do not include this file directly.

#include <utility>

namespace mlinalg {

template <Expression E> constexpr TransposeOp<E>::TransposeOp(E expr) : m_expr(std::move(expr)) {}

template <Expression E> constexpr auto TransposeOp<E>::rows() const -> Index {
    if constexpr (expr_cols_v<E> != Dynamic) {
        return expr_cols_v<E>;
    } else {
        return m_expr.cols();
    }
}

template <Expression E> constexpr auto TransposeOp<E>::cols() const -> Index {
    if constexpr (expr_rows_v<E> != Dynamic) {
        return expr_rows_v<E>;
    } else {
        return m_expr.rows();
    }
}

template <Expression E>
constexpr auto TransposeOp<E>::coeff(Index row, Index col) const -> expr_scalar_t<E> {
    return m_expr.coeff(col, row);
}

template <Expression E> constexpr auto TransposeOp<E>::expression() const noexcept -> const E& {
    return m_expr;
}

} // namespace mlinalg

#endif // AST_TRANSPOSE_OP_IPP
