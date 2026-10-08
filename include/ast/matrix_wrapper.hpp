#ifndef AST_MATRIX_WRAPPER_HPP
#define AST_MATRIX_WRAPPER_HPP

// Module 3: MatrixWrapper - AST leaf referring to a concrete matrix.

#include "expr_traits.hpp"

#include <concepts>

namespace mlinalg {

/// Requirements on a concrete matrix that can appear as an AST leaf. The memory layout is the
/// matrix's own business: the AST only accesses coefficients through m(row, col).
template <typename M>
concept MatrixLeaf = requires {
    { M::RowsAtCompileTime } -> std::convertible_to<int>;
    { M::ColsAtCompileTime } -> std::convertible_to<int>;
    typename M::value_type;
} && requires(const M& matrix, Index row, Index col) {
    { matrix.rows() } -> std::convertible_to<Index>;
    { matrix.cols() } -> std::convertible_to<Index>;
    { matrix(row, col) } -> std::convertible_to<typename M::value_type>;
};

/// Non-owning leaf node. The wrapped matrix must outlive the expression.
template <MatrixLeaf M> class MatrixWrapper {
  public:
    constexpr explicit MatrixWrapper(const M& matrix) noexcept;

    /// Wrapping a temporary would leave the expression dangling.
    MatrixWrapper(const M&&) = delete;

    [[nodiscard]] constexpr auto rows() const -> Index;
    [[nodiscard]] constexpr auto cols() const -> Index;
    [[nodiscard]] constexpr auto coeff(Index row, Index col) const -> M::value_type;

    [[nodiscard]] constexpr auto matrix() const noexcept -> const M&;

  private:
    const M* m_matrix; // Pointer rather than reference so nodes stay copy-assignable.
};

template <MatrixLeaf M> struct expr_traits<MatrixWrapper<M>> {
    static constexpr int Rows = M::RowsAtCompileTime;
    static constexpr int Cols = M::ColsAtCompileTime;
    using ScalarType          = M::value_type;
};

} // namespace mlinalg

#include "matrix_wrapper.ipp" // IWYU pragma: keep

#endif // AST_MATRIX_WRAPPER_HPP
