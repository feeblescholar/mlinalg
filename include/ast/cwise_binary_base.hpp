#ifndef AST_CWISE_BINARY_BASE_HPP
#define AST_CWISE_BINARY_BASE_HPP

// Module 3: shared operand storage and shape logic for coefficient-wise binary nodes
// (AddOp, SubOp). Not a node by itself.

#include "expr_traits.hpp"

namespace mlinalg::detail {

template <Expression L, Expression R>
    requires CwiseCompatible<L, R>
class CwiseBinaryBase {
  public:
    static constexpr int Rows = combine_dims(expr_rows_v<L>, expr_rows_v<R>);
    static constexpr int Cols = combine_dims(expr_cols_v<L>, expr_cols_v<R>);

    /// Operand shapes must match; checked at runtime for Dynamic dimensions.
    constexpr CwiseBinaryBase(L lhs, R rhs);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;

    [[nodiscard]] constexpr auto lhs() const noexcept -> const L&;
    [[nodiscard]] constexpr auto rhs() const noexcept -> const R&;

  private:
    L m_lhs;
    R m_rhs;
};

} // namespace mlinalg::detail

#include "cwise_binary_base.ipp" // IWYU pragma: keep

#endif // AST_CWISE_BINARY_BASE_HPP
