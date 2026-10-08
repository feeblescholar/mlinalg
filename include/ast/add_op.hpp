#ifndef AST_ADD_OP_HPP
#define AST_ADD_OP_HPP

// Module 3: AddOp - coefficient-wise sum of two expressions.

#include "cwise_binary_base.hpp"
#include "expr_traits.hpp"

namespace mlinalg {

template <Expression L, Expression R>
    requires detail::CwiseCompatible<L, R>
class AddOp : public detail::CwiseBinaryBase<L, R> {
  public:
    using detail::CwiseBinaryBase<L, R>::CwiseBinaryBase;

    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> expr_scalar_t<L>;
};

template <typename L, typename R> AddOp(L, R) -> AddOp<L, R>;

template <Expression L, Expression R>
    requires detail::CwiseCompatible<L, R>
struct expr_traits<AddOp<L, R>> {
    static constexpr int Rows = detail::CwiseBinaryBase<L, R>::Rows;
    static constexpr int Cols = detail::CwiseBinaryBase<L, R>::Cols;
    using ScalarType          = expr_scalar_t<L>;
};

} // namespace mlinalg

#include "add_op.ipp" // IWYU pragma: keep

#endif // AST_ADD_OP_HPP
