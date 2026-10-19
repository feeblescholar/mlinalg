#ifndef AST_OWNED_MATRIX_HPP
#define AST_OWNED_MATRIX_HPP

// Module 5: OwnedMatrix - AST leaf owning a matrix that was a temporary operand.
//
// MatrixWrapper refers to its matrix, which would dangle if the matrix were a temporary
// (a * a.inv()). The operators move such a temporary into an OwnedMatrix instead, so the
// expression owns it and stays valid for as long as the expression itself. During evaluation the
// matrix is read through a MatrixWrapper (see eval_tree.hpp), so it is never copied there.

#include "expr_traits.hpp"
#include "matrix_wrapper.hpp"

namespace mlinalg {

/// Owning leaf node. Copying the expression copies the matrix.
template <MatrixLeaf M> class OwnedMatrix {
  public:
    constexpr explicit OwnedMatrix(M matrix);

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> M::value_type;

    [[nodiscard]] constexpr auto matrix() const noexcept -> const M&;

  private:
    M m_matrix;
};

template <MatrixLeaf M> struct expr_traits<OwnedMatrix<M>> {
    static constexpr int Rows = M::RowsAtCompileTime;
    static constexpr int Cols = M::ColsAtCompileTime;
    using ScalarType          = M::value_type;
};

} // namespace mlinalg

#include "owned_matrix.ipp" // IWYU pragma: keep

#endif // AST_OWNED_MATRIX_HPP
