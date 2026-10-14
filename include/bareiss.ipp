#ifndef BAREISS_IPP
#define BAREISS_IPP

// Definitions for bareiss.hpp. Do not include this file directly.

#include "bareiss.hpp" // IWYU pragma: keep

#include <cassert>
#include <utility>

namespace mlinalg::detail {

template <typename M>
    requires BareissScalar<typename M::value_type>
constexpr auto bareiss_determinant(M matrix) -> M::value_type {
    using T = M::value_type;
    assert(matrix.rows() == matrix.cols() && "bareiss_determinant: matrix must be square");

    const auto n = matrix.rows();
    if (n == 0) {
        return T{1};
    }

    // TODO: compute in a wider type (see bareiss.hpp) to delay overflow.
    T    previous_pivot{1};
    bool negate = false;
    for (decltype(matrix.rows()) k = 0; k + 1 < n; ++k) {
        if (matrix(k, k) == T{0}) {
            auto swap_row = k + 1;
            while (swap_row < n && matrix(swap_row, k) == T{0}) {
                ++swap_row;
            }
            if (swap_row == n) {
                return T{0}; // the whole column is zero from row k down
            }
            // Columns before k are no longer read, so only the rest of the rows is swapped.
            for (auto j = k; j < n; ++j) {
                std::swap(matrix(k, j), matrix(swap_row, j));
            }
            negate = !negate;
        }

        const T pivot = matrix(k, k);
        // Column-major: run down the columns. a(i, k) is only read here, never written.
        for (auto j = k + 1; j < n; ++j) {
            const T akj = matrix(k, j);
            for (auto i = k + 1; i < n; ++i) {
                matrix(i, j) = ((matrix(i, j) * pivot) - (matrix(i, k) * akj)) / previous_pivot;
            }
        }
        previous_pivot = pivot;
    }

    const T det = matrix(n - 1, n - 1);
    return negate ? -det : det;
}

template <typename M>
    requires BareissScalar<typename M::value_type>
constexpr auto bareiss_rank(M matrix) -> decltype(matrix.rows()) {
    using T = M::value_type;

    const auto rows = matrix.rows();
    const auto cols = matrix.cols();

    // TODO: compute in a wider type (see bareiss.hpp) to delay overflow.
    T                       previous_pivot{1};
    decltype(matrix.rows()) rank = 0; // also the row the next pivot goes to
    for (decltype(matrix.cols()) col = 0; col < cols && rank < rows; ++col) {
        auto pivot_row = rank;
        while (pivot_row < rows && matrix(pivot_row, col) == T{0}) {
            ++pivot_row;
        }
        if (pivot_row == rows) {
            continue; // no pivot in this column: it depends on the columns before it
        }
        if (pivot_row != rank) {
            // Columns before `col` are no longer read, so only the rest of the rows is swapped.
            for (auto j = col; j < cols; ++j) {
                std::swap(matrix(rank, j), matrix(pivot_row, j));
            }
        }

        const T pivot = matrix(rank, col);
        // Column-major: run down the columns. a(i, col) is only read here, never written.
        for (auto j = col + 1; j < cols; ++j) {
            const T arj = matrix(rank, j);
            for (auto i = rank + 1; i < rows; ++i) {
                matrix(i, j) = ((matrix(i, j) * pivot) - (matrix(i, col) * arj)) / previous_pivot;
            }
        }
        previous_pivot = pivot;
        ++rank;
    }
    return rank;
}

} // namespace mlinalg::detail

#endif // BAREISS_IPP
