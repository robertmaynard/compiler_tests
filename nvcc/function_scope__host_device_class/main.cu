

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/memory.h>
#include <thrust/device_vector.h>

/// Binary Predicate that takes two arguments argument \c x, and \c y and
/// returns True if and only if \c x is equal to \c y.
/// Note: Requires Type \p T implement the == operator.
struct InCompleteWhichWorks {
  __device__ bool operator()(const std::uint8_t &complete) const {
    return !complete;
  }
};

namespace vtkm {

// Unary function object wrapper which can detect and handle calling the
// wrapped operator with complex value types such as
// PortalValue which happen when passed an input array that
// is implicit.
template <typename T_, typename Function> struct WrappedUnaryPredicate {
  using T = typename std::remove_const<T_>::type;

  // make typedefs that thust expects unary operators to have
  using first_argument_type = T;
  using result_type = bool;

  Function m_f;

  __device__ WrappedUnaryPredicate() : m_f() {}

  __host__ WrappedUnaryPredicate(const Function &f) : m_f(f) {}

  __device__ bool operator()(const T &x) const { return m_f(x); }

  __device__ bool operator()(const T *x) const { return m_f(*x); }
};

template <typename InputIterator, typename Stencil, typename OutputIterator,
          typename UnaryPredicate>
__host__ __device__ static std::int64_t
copy_if(InputIterator valuesBegin, InputIterator valuesEnd, Stencil stencil,
        OutputIterator outputBegin, UnaryPredicate unary_predicate) {

  using ValueType = typename Stencil::value_type;

  vtkm::WrappedUnaryPredicate<ValueType, UnaryPredicate> up(unary_predicate);

  auto newLast =
      ::thrust::copy_if(::thrust::cuda::par.on(cudaStreamPerThread),
                        valuesBegin, valuesEnd, stencil, outputBegin, up);
  return static_cast<std::int64_t>(::thrust::distance(outputBegin, newLast));
}

} // namespace vtkm

struct Task {

  __host__ __device__ bool operator()() const {

    struct InComplete {
      __device__ bool operator()(std::uint8_t complete) const {
        return !complete;
      }
    };

    thrust::device_vector<float> points(500);
    thrust::device_vector<std::uint8_t> stencil(500);

    auto size = vtkm::copy_if(points.begin(), points.end(), stencil.begin(),
                              points.begin(), InComplete());
    return size > 0;
  }
};

int main(int argc, char **) {
  Task t;
  return t() ? 0 : 1;
}
