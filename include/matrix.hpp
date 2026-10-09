#ifndef MATRIX_HPP
#define MATRIX_HPP

// Module 4: The Matrix & Vector API - user-facing dense matrix type.
//
// Matrix<T, Rows, Cols> owns its coefficients through MatrixStorage<T, Rows, Cols> (Module 1) and
// stores them in column-major order: coefficient (row, col) lives at linear index
// col * rows() + row. A vector is a matrix with one dimension fixed to 1 (see Vector, RowVector).
//
// Expressions (Module 3) are evaluated only when they are assigned to a Matrix, either through
// the evaluation constructor or through operator=.

#include "ast/expr_traits.hpp"
#include "ast/matrix_wrapper.hpp"
#include "matrix_storage.hpp"

#include <concepts>
#include <initializer_list>
#include <type_traits>

namespace mlinalg {

namespace detail {

/// One dimension is fixed to 1 at compile time.
template <int Rows, int Cols>
concept VectorDims = Rows == 1 || Cols == 1;

/// One dimension is fixed to 1 and the other one is Dynamic.
template <int Rows, int Cols>
concept DynamicVectorDims = (Rows == 1 && Cols == Dynamic) || (Rows == Dynamic && Cols == 1);

/// An expression whose result can be stored in the matrix type M: same scalar type and
/// statically compatible shape. Dynamic dimensions are checked at runtime.
///
/// MatrixWrapper<M> itself is excluded before Expression is checked: Expression requires
/// copy_constructible<MatrixWrapper<M>>, which considers the conversion MatrixWrapper<M> -> M
/// through M's evaluation constructor, which would check EvaluableTo<MatrixWrapper<M>, M> again.
/// Evaluating a matrix's own wrapper is a plain copy anyway.
template <typename E, typename M>
concept EvaluableTo = !std::same_as<E, MatrixWrapper<M>> && Expression<E> &&
                      std::same_as<expr_scalar_t<E>, typename M::value_type> &&
                      dims_compatible(expr_rows_v<E>, M::RowsAtCompileTime) &&
                      dims_compatible(expr_cols_v<E>, M::ColsAtCompileTime);

} // namespace detail

/// Dense, column-major matrix. Rows and Cols are positive compile-time extents or Dynamic.
template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
class Matrix {
  public:
    using value_type      = T;
    using storage_type    = MatrixStorage<T, Rows, Cols>;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = const T*;

    static constexpr int  RowsAtCompileTime = Rows;
    static constexpr int  ColsAtCompileTime = Cols;
    static constexpr bool IsVector          = detail::VectorDims<Rows, Cols>;

    /// Fixed-size matrices are value-initialized. Dynamic dimensions start out as 0.
    constexpr Matrix() = default;

    /// Allocates a rows x cols matrix with value-initialized coefficients. Fixed dimensions must
    /// match their compile-time value.
    constexpr Matrix(Index rows, Index cols)
        requires(!storage_type::is_fixed);

    /// Allocates a vector of `size` value-initialized coefficients.
    constexpr explicit Matrix(Index size)
        requires detail::DynamicVectorDims<Rows, Cols>;

    /// Row-by-row initialization: Matrix m{{1, 2, 3}, {4, 5, 6}} is a 2 x 3 matrix.
    /// Every row must have the same length. Fixed dimensions must match.
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> rows)
        requires(!IsVector);

    /// Vector initialization from a flat list of coefficients: Vector<double> v{1, 2, 3}.
    /// A fixed length must match.
    constexpr Matrix(std::initializer_list<T> coeffs)
        requires IsVector;

    /// Evaluation constructor: computes every coefficient of `expr`.
    /// Implicit so that `Matrix c = a + b;` works.
    template <typename E>
        requires detail::EvaluableTo<E, Matrix>
    constexpr Matrix(const E& expr); // NOLINT(*-explicit-constructor)

    /// Evaluates `expr` and stores the result, resizing Dynamic dimensions as needed.
    /// `expr` may refer to *this. Strong exception guarantee.
    template <typename E>
        requires detail::EvaluableTo<E, Matrix>
    constexpr auto operator=(const E& expr) -> Matrix&;

    [[nodiscard]] constexpr auto rows() const noexcept -> Index;
    [[nodiscard]] constexpr auto cols() const noexcept -> Index;
    [[nodiscard]] constexpr auto size() const noexcept -> Index;

    /// Coefficient at (row, col).
    [[nodiscard]] constexpr auto operator()(Index row, Index col) noexcept -> reference;
    [[nodiscard]] constexpr auto operator()(Index row, Index col) const noexcept -> const_reference;

    /// Vector coefficient at position i.
    [[nodiscard]] constexpr auto operator()(Index i) noexcept -> reference
        requires IsVector;
    [[nodiscard]] constexpr auto operator()(Index i) const noexcept -> const_reference
        requires IsVector;
    [[nodiscard]] constexpr auto operator[](Index i) noexcept -> reference
        requires IsVector;
    [[nodiscard]] constexpr auto operator[](Index i) const noexcept -> const_reference
        requires IsVector;

    /// Contiguous column-major coefficients.
    [[nodiscard]] constexpr auto data() noexcept -> pointer;
    [[nodiscard]] constexpr auto data() const noexcept -> const_pointer;

    /// Iteration over the coefficients in column-major order.
    [[nodiscard]] constexpr auto begin() noexcept -> iterator;
    [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator;
    [[nodiscard]] constexpr auto end() noexcept -> iterator;
    [[nodiscard]] constexpr auto end() const noexcept -> const_iterator;

    /// Changes the Dynamic dimensions; fixed dimensions must keep their compile-time value.
    /// If size() changes, the coefficient values are unspecified afterwards.
    constexpr void resize(Index rows, Index cols);

    /// Changes the length of a Dynamic vector, with the same semantics as resize(rows, cols).
    constexpr void resize(Index size)
        requires detail::DynamicVectorDims<Rows, Cols>;

    constexpr void swap(Matrix& other) noexcept(std::is_nothrow_swappable_v<storage_type>);

    friend constexpr void swap(Matrix& lhs,
                               Matrix& rhs) noexcept(std::is_nothrow_swappable_v<storage_type>) {
        lhs.swap(rhs);
    }

  private:
    [[nodiscard]] constexpr auto index_of(Index row, Index col) const noexcept -> Index;

    storage_type m_storage;
};

/// Dynamic-length column vector.
template <StorageElement T> using Vector = Matrix<T, Dynamic, 1>;

/// Dynamic-length row vector.
template <StorageElement T> using RowVector = Matrix<T, 1, Dynamic>;

} // namespace mlinalg

#include "matrix.ipp" // IWYU pragma: keep

#endif // MATRIX_HPP
