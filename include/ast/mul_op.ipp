#ifndef AST_MUL_OP_IPP
#define AST_MUL_OP_IPP

// Definitions for mul_op.hpp. Do not include this file directly.

#include <cassert>
#include <utility>

namespace mlinalg {

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
constexpr MulOp<L, R>::MulOp(L lhs, R rhs) : m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {
    assert(m_lhs.cols() == m_rhs.rows() && "matrix product with mismatched inner dimensions");
}

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
constexpr auto MulOp<L, R>::rows() const -> Index {
    if constexpr (expr_rows_v<L> != Dynamic) {
        return expr_rows_v<L>;
    } else {
        return m_lhs.rows();
    }
}

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
constexpr auto MulOp<L, R>::cols() const -> Index {
    if constexpr (expr_cols_v<R> != Dynamic) {
        return expr_cols_v<R>;
    } else {
        return m_rhs.cols();
    }
}

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
constexpr auto MulOp<L, R>::coeff(Index row, Index col) const -> expr_scalar_t<L> {
    expr_scalar_t<L> sum{};
    const Index      inner = m_lhs.cols();
    for (Index k = 0; k < inner; ++k) {
        sum += m_lhs.coeff(row, k) * m_rhs.coeff(k, col);
    }
    return sum;
}

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
constexpr auto MulOp<L, R>::lhs() const noexcept -> const L& {
    return m_lhs;
}

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
constexpr auto MulOp<L, R>::rhs() const noexcept -> const R& {
    return m_rhs;
}

} // namespace mlinalg

#endif // AST_MUL_OP_IPP
