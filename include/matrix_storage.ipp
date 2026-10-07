#ifndef MATRIX_STORAGE_IPP
#define MATRIX_STORAGE_IPP

// Definitions for matrix_storage.hpp. Do not include this file directly.

#include "matrix_storage.hpp" // IWYU pragma: keep

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mlinalg {

// ------------------------------------------------------------------------------------------------
// aligned_allocator
// ------------------------------------------------------------------------------------------------

template <typename T, std::size_t Alignment>
auto aligned_allocator<T, Alignment>::allocate(size_type n) -> T* {
    if (n > max_size()) {
        throw std::bad_array_new_length();
    }
    return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t{Alignment}));
}

template <typename T, std::size_t Alignment>
void aligned_allocator<T, Alignment>::deallocate(T* ptr, size_type n) noexcept {
    ::operator delete(ptr, n * sizeof(T), std::align_val_t{Alignment});
}

template <typename T, std::size_t Alignment>
constexpr auto aligned_allocator<T, Alignment>::max_size() noexcept -> size_type {
    return std::numeric_limits<size_type>::max() / sizeof(T);
}

// ------------------------------------------------------------------------------------------------
// FixedStorage
// ------------------------------------------------------------------------------------------------

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr FixedStorage<T, Rows, Cols>::FixedStorage([[maybe_unused]] Index rows,
                                                    [[maybe_unused]] Index cols) {
    assert(rows == Rows && cols == Cols && "FixedStorage: dimensions must match Rows and Cols");
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::rows() noexcept -> Index {
    return Rows;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::cols() noexcept -> Index {
    return Cols;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::size() noexcept -> Index {
    return static_cast<Index>(kSize);
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::data() noexcept -> pointer {
    return m_data.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::data() const noexcept -> const_pointer {
    return m_data.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::operator[](Index i) noexcept -> reference {
    assert(i >= 0 && i < size() && "FixedStorage: index out of range");
    return m_data[static_cast<std::size_t>(i)];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::operator[](Index i) const noexcept -> const_reference {
    assert(i >= 0 && i < size() && "FixedStorage: index out of range");
    return m_data[static_cast<std::size_t>(i)];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::begin() noexcept -> iterator {
    return m_data.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::begin() const noexcept -> const_iterator {
    return m_data.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::end() noexcept -> iterator {
    return m_data.data() + kSize;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr auto FixedStorage<T, Rows, Cols>::end() const noexcept -> const_iterator {
    return m_data.data() + kSize;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr void FixedStorage<T, Rows, Cols>::resize([[maybe_unused]] Index rows,
                                                   [[maybe_unused]] Index cols) {
    assert(rows == Rows && cols == Cols && "FixedStorage: cannot resize a fixed-size matrix");
}

template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
constexpr void
FixedStorage<T, Rows, Cols>::swap(FixedStorage& other) noexcept(std::is_nothrow_swappable_v<T>) {
    m_data.swap(other.m_data);
}

// ------------------------------------------------------------------------------------------------
// DynamicStorage
// ------------------------------------------------------------------------------------------------

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
DynamicStorage<T, Rows, Cols>::DynamicStorage(Index rows, Index cols) : m_rows(rows), m_cols(cols) {
    const Index n = checked_size(rows, cols);
    if (n > SboCapacity) {
        m_heap = allocate_elements(n);
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
DynamicStorage<T, Rows, Cols>::DynamicStorage(const DynamicStorage& other)
    : m_rows(other.m_rows), m_cols(other.m_cols) {
    if (other.m_heap != nullptr) {
        m_heap = allocate_copy(other.m_heap, other.size());
    } else {
        std::copy_n(other.m_sbo.begin(), other.size(), m_sbo.begin());
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
DynamicStorage<T, Rows, Cols>::DynamicStorage(DynamicStorage&& other) noexcept(
    std::is_nothrow_move_assignable_v<T>) {
    steal(other);
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::operator=(const DynamicStorage& other) -> DynamicStorage& {
    if (this == &other) {
        return *this;
    }
    if (size() != other.size()) {
        // Allocate before releasing so a throwing allocation leaves *this untouched.
        T* new_heap = other.m_heap != nullptr ? allocate_copy(other.m_heap, other.size()) : nullptr;
        release_heap();
        m_heap = new_heap;
        if (m_heap == nullptr) {
            std::copy_n(other.m_sbo.begin(), other.size(), m_sbo.begin());
        }
    } else {
        std::copy_n(other.data(), other.size(), data());
    }
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    return *this;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::operator=(DynamicStorage&& other) noexcept(
    std::is_nothrow_move_assignable_v<T>) -> DynamicStorage& {
    if (this != &other) {
        release_heap();
        steal(other);
    }
    return *this;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
DynamicStorage<T, Rows, Cols>::~DynamicStorage() {
    release_heap();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::rows() const noexcept -> Index {
    return m_rows;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::cols() const noexcept -> Index {
    return m_cols;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::size() const noexcept -> Index {
    return m_rows * m_cols;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::is_heap_allocated() const noexcept -> bool {
    return m_heap != nullptr;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::data() noexcept -> pointer {
    return m_heap != nullptr ? m_heap : m_sbo.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::data() const noexcept -> const_pointer {
    return m_heap != nullptr ? m_heap : m_sbo.data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::operator[](Index i) noexcept -> reference {
    assert(i >= 0 && i < size() && "DynamicStorage: index out of range");
    return data()[i];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::operator[](Index i) const noexcept -> const_reference {
    assert(i >= 0 && i < size() && "DynamicStorage: index out of range");
    return data()[i];
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::begin() noexcept -> iterator {
    return data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::begin() const noexcept -> const_iterator {
    return data();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::end() noexcept -> iterator {
    return data() + size();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::end() const noexcept -> const_iterator {
    return data() + size();
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
void DynamicStorage<T, Rows, Cols>::resize(Index rows, Index cols) {
    const Index n = checked_size(rows, cols);
    if (n != size()) {
        // Allocate before releasing so a throwing allocation leaves *this untouched.
        T* new_heap = n > SboCapacity ? allocate_elements(n) : nullptr;
        release_heap();
        m_heap = new_heap;
    }
    m_rows = rows;
    m_cols = cols;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
void DynamicStorage<T, Rows, Cols>::swap(DynamicStorage& other) noexcept(
    std::is_nothrow_swappable_v<T>) {
    // Only the inline elements that are actually in use need to be exchanged.
    const Index sbo_used =
        std::max(m_heap == nullptr ? size() : 0, other.m_heap == nullptr ? other.size() : 0);
    std::swap_ranges(m_sbo.begin(), m_sbo.begin() + sbo_used, other.m_sbo.begin());
    std::swap(m_rows, other.m_rows);
    std::swap(m_cols, other.m_cols);
    std::swap(m_heap, other.m_heap);
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::checked_size(Index rows, Index cols) -> Index {
    assert(rows >= 0 && cols >= 0 && "DynamicStorage: negative dimension");
    assert((Rows == Dynamic || rows == Rows) &&
           "DynamicStorage: row count is fixed at compile time");
    assert((Cols == Dynamic || cols == Cols) &&
           "DynamicStorage: col count is fixed at compile time");
    if (cols != 0 && rows > std::numeric_limits<Index>::max() / cols) {
        throw std::length_error("mlinalg::DynamicStorage: rows * cols overflows Index");
    }
    return rows * cols;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::allocate_elements(Index n) -> T* {
    T* ptr = m_alloc.allocate(static_cast<std::size_t>(n));
    try {
        std::uninitialized_value_construct_n(ptr, n);
    } catch (...) {
        m_alloc.deallocate(ptr, static_cast<std::size_t>(n));
        throw;
    }
    return ptr;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
auto DynamicStorage<T, Rows, Cols>::allocate_copy(const T* src, Index n) -> T* {
    T* ptr = m_alloc.allocate(static_cast<std::size_t>(n));
    try {
        std::uninitialized_copy_n(src, n, ptr);
    } catch (...) {
        m_alloc.deallocate(ptr, static_cast<std::size_t>(n));
        throw;
    }
    return ptr;
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
void DynamicStorage<T, Rows, Cols>::release_heap() noexcept {
    if (m_heap != nullptr) {
        std::destroy_n(m_heap, size());
        m_alloc.deallocate(m_heap, static_cast<std::size_t>(size()));
        m_heap = nullptr;
    }
}

template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
void DynamicStorage<T, Rows, Cols>::steal(DynamicStorage& other) noexcept(
    std::is_nothrow_move_assignable_v<T>) {
    assert(m_heap == nullptr && "DynamicStorage: steal() requires released heap storage");
    if (other.m_heap != nullptr) {
        m_heap = std::exchange(other.m_heap, nullptr);
    } else {
        std::move(other.m_sbo.begin(), other.m_sbo.begin() + other.size(), m_sbo.begin());
    }
    m_rows = std::exchange(other.m_rows, kInitialRows);
    m_cols = std::exchange(other.m_cols, kInitialCols);
}

} // namespace mlinalg

#endif // MATRIX_STORAGE_IPP
