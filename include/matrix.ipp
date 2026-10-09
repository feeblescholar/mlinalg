#ifndef MATRIX_IPP
#define MATRIX_IPP

// Definitions for matrix.hpp. Do not include this file directly.

#include "matrix.hpp" // IWYU pragma: keep

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <type_traits>
#include <utility>

namespace mlinalg {

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   constexpr Matrix<T, Rows, Cols>::Matrix(Index rows, Index cols)
                 requires(!storage_type::is_fixed)
    : m_storage(rows, cols) {}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   constexpr Matrix<T, Rows, Cols>::Matrix(Index size)
                 requires detail::DynamicVectorDims<Rows, Cols>
    : m_storage(Rows == 1 ? 1 : size, Rows == 1 ? size : 1) {}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   constexpr Matrix<T, Rows, Cols>::Matrix(
                                       std::initializer_list<std::initializer_list<T>> rows)
                 requires(!IsVector)
    : m_storage(static_cast<Index>(rows.size()), Cols != Dynamic ? Cols
                                                 : rows.size() == 0
                                                     ? 0
                                                     : static_cast<Index>(rows.begin()->size())) {
    Index row = 0;
    for (const std::initializer_list<T>& coeffs : rows) {
        assert(static_cast<Index>(coeffs.size()) == cols() &&
               "Matrix: initializer list rows must all have cols() coefficients");
        Index col = 0;
        for (const T& coeff : coeffs) {
            m_storage[index_of(row, col++)] = coeff;
        }
        ++row;
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> &&
             detail::ValidDim<Cols>
             constexpr Matrix<T, Rows, Cols>::Matrix(std::initializer_list<T> coeffs)
                 requires IsVector
    : m_storage(Rows == 1 ? 1 : static_cast<Index>(coeffs.size()),
                Rows == 1 ? static_cast<Index>(coeffs.size()) : 1) {
    std::ranges::copy(coeffs, m_storage.begin());
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   template <typename E>
                 requires detail::EvaluableTo<E, Matrix<T, Rows, Cols>>
constexpr Matrix<T, Rows, Cols>::Matrix(const E& expr) : m_storage(expr.rows(), expr.cols()) {
    // Naive coefficient-wise evaluation; Module 5 replaces it with the vectorized evaluator.
    for (Index col = 0; col < cols(); ++col) {
        for (Index row = 0; row < rows(); ++row) {
            m_storage[index_of(row, col)] = expr.coeff(row, col);
        }
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   template <typename E>
                 requires detail::EvaluableTo<E, Matrix<T, Rows, Cols>>
constexpr auto Matrix<T, Rows, Cols>::operator=(const E& expr) -> Matrix& {
    // Evaluating into a temporary keeps aliasing expressions (a = a * b) correct, and leaves
    // *this untouched if evaluation throws.
    Matrix result(expr);
    m_storage = std::move(result.m_storage);
    return *this;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::rows() const noexcept -> Index {
    return m_storage.rows();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::cols() const noexcept -> Index {
    return m_storage.cols();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::size() const noexcept -> Index {
    return m_storage.size();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::operator()(Index row, Index col) noexcept -> reference {
    return m_storage[index_of(row, col)];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::operator()(Index row,
                                                 Index col) const noexcept -> const_reference {
    return m_storage[index_of(row, col)];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
             constexpr auto Matrix<T, Rows, Cols>::operator()(Index i) noexcept -> reference
                 requires IsVector
{
    return m_storage[i];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
             constexpr auto
             Matrix<T, Rows, Cols>::operator()(Index i) const noexcept -> const_reference
                 requires IsVector
{
    return m_storage[i];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
             constexpr auto Matrix<T, Rows, Cols>::operator[](Index i) noexcept -> reference
                 requires IsVector
{
    return m_storage[i];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
             constexpr auto
             Matrix<T, Rows, Cols>::operator[](Index i) const noexcept -> const_reference
                 requires IsVector
{
    return m_storage[i];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::data() noexcept -> pointer {
    return m_storage.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::data() const noexcept -> const_pointer {
    return m_storage.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::begin() noexcept -> iterator {
    return m_storage.begin();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::begin() const noexcept -> const_iterator {
    return m_storage.begin();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::end() noexcept -> iterator {
    return m_storage.end();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::end() const noexcept -> const_iterator {
    return m_storage.end();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr void Matrix<T, Rows, Cols>::resize(Index rows, Index cols) {
    m_storage.resize(rows, cols);
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
                                   constexpr void Matrix<T, Rows, Cols>::resize(Index size)
                 requires detail::DynamicVectorDims<Rows, Cols>
{
    if constexpr (Rows == 1) {
        m_storage.resize(1, size);
    } else {
        m_storage.resize(size, 1);
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr void
Matrix<T, Rows, Cols>::swap(Matrix& other) noexcept(std::is_nothrow_swappable_v<storage_type>) {
    using std::swap;
    swap(m_storage, other.m_storage);
}

template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
constexpr auto Matrix<T, Rows, Cols>::index_of(Index row, Index col) const noexcept -> Index {
    assert(row >= 0 && row < rows() && col >= 0 && col < cols() &&
           "Matrix: coefficient index out of range");
    return (col * rows()) + row;
}

} // namespace mlinalg

#endif // MATRIX_IPP
