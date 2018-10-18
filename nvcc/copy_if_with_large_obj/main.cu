#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include <iostream>

template <typename T1, typename T2, typename P>
void ArrayCompact(thrust::device_vector<T1>& array,
                  const thrust::device_vector<T2>& stencil,
                  const P& predicate)
{
  thrust::device_vector<T1> result(array.size());
  auto newEnd = thrust::copy_if(array.begin(),
                                array.end(),
                                stencil.begin(),
                                result.begin(),
                                predicate);
  result.resize(thrust::distance(result.begin(), newEnd));
  array.swap(result);
}

template <typename Container>
void PrintArray(const Container& array, std::ostream& out)
{
  out << "size: " << array.size() << ", vals: ";
  for (const auto& e : array)
  {
    out << e << " ";
  }
  out << "\n";
}

struct Element
{
  static constexpr int count = 16;
  std::int64_t data[count];
};

std::ostream& operator<<(std::ostream& out, const Element& e)
{
  out << "[" << e.data[0];
  for (int i = 1; i < Element::count; ++i)
  {
    out << "," << e.data[i];
  }
  out << "]";
  return out;
}

struct CopyPredicate
{
  __host__ __device__ bool operator()(int val) const
  {
    return val == 0;
  }
};

int main()
{
  std::cout << "sizeof(Element) = " << sizeof(Element) << "\n";

  thrust::host_vector<Element> hostArray(5);
  for (int i = 0; i < 5; ++i)
  {
    for (int j = 0; j < Element::count; ++j)
    {
      hostArray[i].data[j] = (i * Element::count) + j;
    }
  }
  PrintArray(hostArray, std::cout);

  int stencil_data[] = { 0, 1, 0, 1, 0 };
  thrust::device_vector<int> stencil(5);
  thrust::copy(stencil_data, stencil_data + 5, stencil.begin());

  thrust::device_vector<Element> deviceArray;

  deviceArray = hostArray;
  ArrayCompact(deviceArray, stencil, CopyPredicate());
  hostArray = deviceArray;
  PrintArray(hostArray, std::cout);
}
