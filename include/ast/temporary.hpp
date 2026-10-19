#ifndef AST_TEMPORARY_HPP
#define AST_TEMPORARY_HPP

// Module 5: Temporary - AST leaf owning the evaluated coefficients of an expression.
//
// The one exception to "no temporaries": a matrix product reads every operand coefficient many
// times, so an operand that is itself an expression is evaluated once into a Temporary instead of
// being recomputed on every access (see eval_tree.hpp). Never created by the user-facing operators.

#include "expr_traits.hpp"

namespace mlinalg {

/// Owning leaf node holding a column-major copy of an evaluated expression.
template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
class Temporary {
  public:
    /// Evaluates every coefficient of `expr` once. Temporary itself is excluded before Expression
    /// is checked: Expression requires copy_constructible, which would consider this constructor
    /// with E = Temporary and recurse (the same issue as detail::EvaluableTo in matrix.hpp).
    template <typename E>
        requires(!std::same_as<E, Temporary<T, Rows, Cols>>) && Expression<E> &&
                std::same_as<expr_scalar_t<E>, T> && (expr_rows_v<E> == Rows) &&
                (expr_cols_v<E> == Cols)
    constexpr explicit Temporary(const E& expr);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> T;

  private:
    MatrixStorage<T, Rows, Cols> m_storage;
};

template <StorageElement T, int R, int C>
    requires detail::ValidDim<R> && detail::ValidDim<C>
struct expr_traits<Temporary<T, R, C>> {
    static constexpr int Rows = R;
    static constexpr int Cols = C;
    using ScalarType          = T;
};

/// The Temporary type that holds the result of expression E.
template <Expression E>
using temporary_for_t = Temporary<expr_scalar_t<E>, expr_rows_v<E>, expr_cols_v<E>>;

} // namespace mlinalg

#include "temporary.ipp" // IWYU pragma: keep

#endif // AST_TEMPORARY_HPP
