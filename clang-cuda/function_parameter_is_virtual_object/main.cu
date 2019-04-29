
#include <thrust/system/cuda/vector.h>

#define HOST_DEVICE __attribute__((device)) __attribute__((host))
namespace vtkm {

template <typename T, int Size> class Vec {
public:
  HOST_DEVICE constexpr Vec() {}
  HOST_DEVICE constexpr Vec(const T &x, const T &y, const T &z) {}

  T m_data[Size];
};

template <typename T>
class ArrayPortalVirtual {
public:
  using ValueType = T;
  HOST_DEVICE virtual T Value(int index) const = 0;
};
template <typename T> class ArrayPortalRef {
public:
  using ValueType = T;
  HOST_DEVICE inline T Get(int index) const {
    return this->Portal->Value(index);
  }
  const ArrayPortalVirtual<T> *Portal = nullptr;
};

namespace exec {
class CellLocator {
public:
  HOST_DEVICE virtual int FindCell(const ArrayPortalRef<vtkm::Vec<float, 3>> &coords) const = 0;
};

class CellLocatorUniformGrid : public vtkm::exec::CellLocator {
private:
public:
  HOST_DEVICE CellLocatorUniformGrid() {}

  HOST_DEVICE virtual int
  FindCell(const ArrayPortalRef<vtkm::Vec<float, 3>> &coords) const override {
    return coords.Get(0).m_data[0];
  }
};

template <typename VirtualDerivedType>
__global__ void ConstructVirtualObjectKernel(VirtualDerivedType **deviceObject) {

  *deviceObject = new VirtualDerivedType();
}
void allocate_on_device() {
  CellLocatorUniformGrid **execGrid;
  cudaMalloc(&execGrid, sizeof(CellLocatorUniformGrid **));
  ConstructVirtualObjectKernel<<<1, 1, 0, ((cudaStream_t)0x2)>>>(execGrid);

}

} // namespace exec
} // namespace vtkm
