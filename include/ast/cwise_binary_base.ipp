#ifndef AST_CWISE_BINARY_BASE_IPP
#define AST_CWISE_BINARY_BASE_IPP

// Definitions for cwise_binary_base.hpp. Do not include this file directly.

#include <cassert>
#include <utility>

namespace mlinalg::detail {

template <Expression L, Expression R>
    requires CwiseCompatible<L, R>
constexpr CwiseBinaryBase<L, R>::CwiseBinaryBase(L lhs, R rhs)
    : m_lhs(std::move(lhs)), m_rhs(std::move(rhs)) {
    assert(m_lhs.rows() == m_rhs.rows() && m_lhs.cols() == m_rhs.cols() &&
           "coefficient-wise operation on matrices of different shapes");
}

template <Expression L, Expression R>
    requires CwiseCompatible<L, R>
constexpr auto CwiseBinaryBase<L, R>::rows() const -> Index {
    if constexpr (Rows != Dynamic) {
        return Rows;
    } else {
        return m_lhs.rows();
    }
}

template <Expression L, Expression R>
    requires CwiseCompatible<L, R>
constexpr auto CwiseBinaryBase<L, R>::cols() const -> Index {
    if constexpr (Cols != Dynamic) {
        return Cols;
    } else {
        return m_lhs.cols();
    }
}

template <Expression L, Expression R>
    requires CwiseCompatible<L, R>
constexpr auto CwiseBinaryBase<L, R>::lhs() const noexcept -> const L& {
    return m_lhs;
}

template <Expression L, Expression R>
    requires CwiseCompatible<L, R>
constexpr auto CwiseBinaryBase<L, R>::rhs() const noexcept -> const R& {
    return m_rhs;
}

} // namespace mlinalg::detail

#endif // AST_CWISE_BINARY_BASE_IPP
