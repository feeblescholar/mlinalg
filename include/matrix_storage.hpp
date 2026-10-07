#ifndef MATRIX_STORAGE_HPP
#define MATRIX_STORAGE_HPP

// Module 1: Memory & Storage Policies.
//
// Provides 32-byte aligned storage for matrix coefficients:
//   * aligned_allocator<T, Alignment> - used for every heap allocation in the library.
//   * FixedStorage<T, Rows, Cols>     - both dimensions known at compile time; inline buffer.
//   * DynamicStorage<T, Rows, Cols>   - at least one Dynamic dimension; small buffer
//                                       optimization (SBO) for up to SboCapacity elements.
//   * MatrixStorage<T, Rows, Cols>    - selects one of the two policies above.
//
// The storage layer is layout-agnostic: it exposes a contiguous block of rows() * cols()
// elements. Mapping (row, col) to a linear index is the responsibility of the matrix frontend.

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace mlinalg {

using Index = std::ptrdiff_t;

/// Marker for a dimension whose extent is only known at runtime.
inline constexpr int Dynamic = -1;

/// Minimum alignment (in bytes) of all coefficient storage, required for aligned AVX2 loads.
inline constexpr std::size_t DefaultAlignment = 32;

/// Maximum number of elements a DynamicStorage keeps in its inline buffer.
inline constexpr Index SboCapacity = 32;

/// Alignment actually used for storing T: DefaultAlignment, or stricter if T demands it.
template <typename T>
inline constexpr std::size_t storage_alignment_v = std::max(DefaultAlignment, alignof(T));

/// Requirements on matrix coefficient types.
template <typename T>
concept StorageElement =
    std::is_object_v<T> && !std::is_const_v<T> && std::default_initializable<T> && std::copyable<T>;

namespace detail {

template <int Dim>
concept ValidDim = Dim == Dynamic || Dim > 0;

template <int Rows, int Cols>
concept FixedDims = Rows > 0 && Cols > 0;

template <int Rows, int Cols>
concept DynamicDims = ValidDim<Rows> && ValidDim<Cols> && (Rows == Dynamic || Cols == Dynamic);

} // namespace detail

// ------------------------------------------------------------------------------------------------
// aligned_allocator
// ------------------------------------------------------------------------------------------------

/// Stateless, standard-conforming allocator returning memory aligned to `Alignment` bytes.
template <typename T, std::size_t Alignment = storage_alignment_v<T>> class aligned_allocator {
    static_assert(std::has_single_bit(Alignment), "Alignment must be a power of two");
    static_assert(Alignment >= alignof(T), "Alignment must not be weaker than alignof(T)");

  public:
    using value_type                             = T;
    using size_type                              = std::size_t;
    using difference_type                        = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal                        = std::true_type;

    static constexpr std::size_t alignment = Alignment;

    template <typename U> struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    constexpr aligned_allocator() noexcept = default;

    template <typename U>
    constexpr explicit aligned_allocator(
        const aligned_allocator<U, Alignment>& /*other*/) noexcept {}

    /// Allocates uninitialized storage for `n` objects of type T.
    /// Throws std::bad_array_new_length on overflow and std::bad_alloc on failure.
    [[nodiscard]] auto allocate(size_type n) -> T*;

    /// Releases storage previously obtained from allocate(n).
    void deallocate(T* ptr, size_type n) noexcept;

    [[nodiscard]] static constexpr auto max_size() noexcept -> size_type;

    template <typename U>
    friend constexpr auto operator==(const aligned_allocator&,
                                     const aligned_allocator<U, Alignment>&) noexcept -> bool {
        return true;
    }
};

// ------------------------------------------------------------------------------------------------
// FixedStorage
// ------------------------------------------------------------------------------------------------

/// Storage for matrices whose dimensions are both known at compile time.
/// Elements live in an aligned inline buffer and are value-initialized on construction.
/// Fully usable in constant expressions.
template <StorageElement T, int Rows, int Cols>
    requires detail::FixedDims<Rows, Cols>
class FixedStorage {
  public:
    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = const T*;

    static constexpr int         RowsAtCompileTime = Rows;
    static constexpr int         ColsAtCompileTime = Cols;
    static constexpr bool        is_fixed          = true;
    static constexpr std::size_t alignment         = storage_alignment_v<T>;

    constexpr FixedStorage() = default;

    /// Provided for uniformity with DynamicStorage; the arguments must equal Rows and Cols.
    constexpr FixedStorage(Index rows, Index cols);

    [[nodiscard]] static constexpr auto rows() noexcept -> Index;
    [[nodiscard]] static constexpr auto cols() noexcept -> Index;
    [[nodiscard]] static constexpr auto size() noexcept -> Index;

    [[nodiscard]] constexpr auto data() noexcept -> pointer;
    [[nodiscard]] constexpr auto data() const noexcept -> const_pointer;

    [[nodiscard]] constexpr auto operator[](Index i) noexcept -> reference;
    [[nodiscard]] constexpr auto operator[](Index i) const noexcept -> const_reference;

    [[nodiscard]] constexpr auto begin() noexcept -> iterator;
    [[nodiscard]] constexpr auto begin() const noexcept -> const_iterator;
    [[nodiscard]] constexpr auto end() noexcept -> iterator;
    [[nodiscard]] constexpr auto end() const noexcept -> const_iterator;

    /// No-op for fixed storage; the arguments must equal Rows and Cols.
    constexpr void resize(Index rows, Index cols);

    constexpr void swap(FixedStorage& other) noexcept(std::is_nothrow_swappable_v<T>);

    friend constexpr void swap(FixedStorage& lhs,
                               FixedStorage& rhs) noexcept(std::is_nothrow_swappable_v<T>) {
        lhs.swap(rhs);
    }

  private:
    static constexpr std::size_t kSize = static_cast<std::size_t>(Rows) * Cols;
    static_assert(kSize / static_cast<std::size_t>(Cols) == static_cast<std::size_t>(Rows),
                  "Rows * Cols overflows");

    alignas(alignment) std::array<T, kSize> m_data{};
};

// ------------------------------------------------------------------------------------------------
// DynamicStorage
// ------------------------------------------------------------------------------------------------

/// Storage for matrices with at least one Dynamic dimension.
///
/// If rows() * cols() <= SboCapacity, elements live in an aligned inline buffer and no heap
/// allocation happens. Larger matrices are allocated through aligned_allocator.
/// A dimension that is fixed at compile time can never change at runtime.
///
/// Newly constructed storage is value-initialized. After a resize() that changes size(),
/// element values are unspecified (but every element is a valid object).
template <StorageElement T, int Rows, int Cols>
    requires detail::DynamicDims<Rows, Cols>
class DynamicStorage {
  public:
    using value_type      = T;
    using allocator_type  = aligned_allocator<T>;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = const T*;

    static constexpr int         RowsAtCompileTime = Rows;
    static constexpr int         ColsAtCompileTime = Cols;
    static constexpr bool        is_fixed          = false;
    static constexpr std::size_t alignment         = storage_alignment_v<T>;

    /// Empty storage: Dynamic dimensions are 0, fixed dimensions keep their compile-time value.
    DynamicStorage() = default;

    /// Allocates rows * cols value-initialized elements. Fixed dimensions must match.
    /// Throws std::length_error if rows * cols overflows Index.
    DynamicStorage(Index rows, Index cols);

    DynamicStorage(const DynamicStorage& other);
    DynamicStorage(DynamicStorage&& other) noexcept(std::is_nothrow_move_assignable_v<T>);
    auto operator=(const DynamicStorage& other) -> DynamicStorage&;
    auto operator=(DynamicStorage&& other) noexcept(std::is_nothrow_move_assignable_v<T>)
        -> DynamicStorage&;
    ~DynamicStorage();

    [[nodiscard]] auto rows() const noexcept -> Index;
    [[nodiscard]] auto cols() const noexcept -> Index;
    [[nodiscard]] auto size() const noexcept -> Index;

    /// True if the elements live on the heap rather than in the inline SBO buffer.
    [[nodiscard]] auto is_heap_allocated() const noexcept -> bool;

    [[nodiscard]] auto data() noexcept -> pointer;
    [[nodiscard]] auto data() const noexcept -> const_pointer;

    [[nodiscard]] auto operator[](Index i) noexcept -> reference;
    [[nodiscard]] auto operator[](Index i) const noexcept -> const_reference;

    [[nodiscard]] auto begin() noexcept -> iterator;
    [[nodiscard]] auto begin() const noexcept -> const_iterator;
    [[nodiscard]] auto end() noexcept -> iterator;
    [[nodiscard]] auto end() const noexcept -> const_iterator;

    /// Changes the dimensions. Fixed dimensions must match. If size() is unchanged the elements
    /// are kept, otherwise their values are unspecified. Strong exception guarantee.
    void resize(Index rows, Index cols);

    void swap(DynamicStorage& other) noexcept(std::is_nothrow_swappable_v<T>);

    friend void swap(DynamicStorage& lhs,
                     DynamicStorage& rhs) noexcept(std::is_nothrow_swappable_v<T>) {
        lhs.swap(rhs);
    }

  private:
    static constexpr Index kInitialRows = Rows == Dynamic ? 0 : Rows;
    static constexpr Index kInitialCols = Cols == Dynamic ? 0 : Cols;

    [[nodiscard]] static auto checked_size(Index rows, Index cols) -> Index;
    [[nodiscard]] auto        allocate_elements(Index n) -> T*;
    [[nodiscard]] auto        allocate_copy(const T* src, Index n) -> T*;
    void                      release_heap() noexcept;
    void steal(DynamicStorage& other) noexcept(std::is_nothrow_move_assignable_v<T>);

    Index m_rows = kInitialRows;
    Index m_cols = kInitialCols;
    T*    m_heap = nullptr; // Non-null iff size() > SboCapacity.
    alignas(alignment) std::array<T, SboCapacity> m_sbo{};
    [[no_unique_address]] allocator_type m_alloc{};
};

// ------------------------------------------------------------------------------------------------
// MatrixStorage
// ------------------------------------------------------------------------------------------------

namespace detail {

// Lazy selection: std::conditional_t would name both specializations, and naming a constrained
// template with unsatisfied constraints is ill-formed.
template <typename T, int Rows, int Cols, bool Fixed = FixedDims<Rows, Cols>>
struct storage_selector {
    using type = DynamicStorage<T, Rows, Cols>;
};

template <typename T, int Rows, int Cols> struct storage_selector<T, Rows, Cols, true> {
    using type = FixedStorage<T, Rows, Cols>;
};

} // namespace detail

/// Storage policy selector: FixedStorage when both dimensions are positive compile-time
/// constants, DynamicStorage when at least one dimension is Dynamic.
template <StorageElement T, int Rows, int Cols>
    requires detail::ValidDim<Rows> && detail::ValidDim<Cols>
using MatrixStorage = typename detail::storage_selector<T, Rows, Cols>::type;

} // namespace mlinalg

#include "matrix_storage.ipp" // IWYU pragma: keep

#endif // MATRIX_STORAGE_HPP
