#ifndef AST_MUL_OP_HPP
#define AST_MUL_OP_HPP

// Module 3: MulOp - matrix product of two expressions.

#include "expr_traits.hpp"

namespace mlinalg {

/// Matrix product lhs * rhs. coeff(i, j) is the dot product of row i of lhs and column j of rhs,
/// i.e. it costs lhs.cols() multiply-adds per coefficient.
template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
class MulOp {
  public:
    /// lhs.cols() must equal rhs.rows(); checked at runtime for Dynamic dimensions.
    constexpr MulOp(L lhs, R rhs);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> expr_scalar_t<L>;

    [[nodiscard]] constexpr auto lhs() const noexcept -> const L&;
    [[nodiscard]] constexpr auto rhs() const noexcept -> const R&;

  private:
    L m_lhs;
    R m_rhs;
};

template <typename L, typename R> MulOp(L, R) -> MulOp<L, R>;

template <Expression L, Expression R>
    requires detail::ProductCompatible<L, R>
struct expr_traits<MulOp<L, R>> {
    static constexpr int Rows = expr_rows_v<L>;
    static constexpr int Cols = expr_cols_v<R>;
    using ScalarType          = expr_scalar_t<L>;
};

} // namespace mlinalg

#include "mul_op.ipp" // IWYU pragma: keep

#endif // AST_MUL_OP_HPP
