#include <iostream>
#include "cuda.h"

//-----------------------------------------------------------------------------
template<class... T> struct list{};

//-----------------------------------------------------------------------------
static inline __host__ __device__ int SignBit(float x)
{
  return static_cast<int>(signbit(x));
}

static inline __host__ __device__ int SignBit(double x)
{
  return static_cast<int>(signbit(x));
}


//-----------------------------------------------------------------------------
template <typename Functor, typename T1, typename T2>
void ListForEach(Functor&& f, list<T1, T2>)
{
  //Fails
  f(T1());
  f(T2());

  // //Works
  // f(T1{});
  // f(T2{});
}

//-----------------------------------------------------------------------------
class ErrorMessageBuffer
{
public:
  __host__ __device__ ErrorMessageBuffer()
    : MessageBuffer(nullptr)
    , MessageBufferSize(0)
  {
  }

  __host__ __device__
  ErrorMessageBuffer(char* messageBuffer, std::size_t bufferSize)
    : MessageBuffer(messageBuffer)
    , MessageBufferSize(bufferSize)
  {
  }

  __host__ __device__ void RaiseError(const char* message) const
  {
    // Only raise the error if one has not been raised yet. This check is not
    // guaranteed to work across threads. However, chances are that if two or
    // more threads simultaneously pass this test, they will be writing the
    // same error, which is fine. Even in the much less likely case that two
    // threads simultaneously write different error messages, the worst case is
    // that you get a mangled message. That's not good (and it's what we are
    // trying to avoid), but it's not critical.
    if (this->IsErrorRaised())
    {
      return;
    }

    // Safely copy message into array.
    for (std::size_t index = 0; index < this->MessageBufferSize; index++)
    {
      this->MessageBuffer[index] = message[index];
      if (message[index] == '\0')
      {
        break;
      }
    }

    // Make sure message is null terminated.
    this->MessageBuffer[this->MessageBufferSize - 1] = '\0';
  }

  __host__ __device__ bool IsErrorRaised() const
  {
    if (this->MessageBufferSize > 0)
    {
      return (this->MessageBuffer[0] != '\0');
    }
    else
    {
      // If there is no buffer set, then always report an error.
      return true;
    }
  }

private:
  char* MessageBuffer;
  std::size_t MessageBufferSize;
};

//-----------------------------------------------------------------------------
class FunctorBase
{
public:
  __host__ __device__
  FunctorBase()
    : ErrorMessage()
  {
  }

  __host__ __device__
  void RaiseError(const char* message) const { this->ErrorMessage.RaiseError(message); }

  __host__
  void SetErrorMessageBuffer(const ErrorMessageBuffer& buffer)
  {
    this->ErrorMessage = buffer;
  }

private:
  ErrorMessageBuffer ErrorMessage;
};

//-----------------------------------------------------------------------------
template <class FunctorType>
__global__ void Schedule1DIndexKernel(FunctorType functor,
                                      std::int64_t numberOfKernelsInvoked,
                                      std::int64_t length)
{
  const std::int64_t index =
    numberOfKernelsInvoked + static_cast<std::size_t>(blockDim.x * blockIdx.x + threadIdx.x);
  if (index < length)
  {
    functor(index);
  }
}

//-----------------------------------------------------------------------------
template <class Functor>
static void Schedule(Functor functor, std::int64_t numInstances)
{
  std::size_t errorArraySize = 1024;
  char *hostErrorPtr = nullptr;
  char* deviceErrorPtr = nullptr;

  cudaMallocHost((void**)&hostErrorPtr, errorArraySize, cudaHostAllocMapped);
  cudaHostGetDevicePointer(&deviceErrorPtr, hostErrorPtr, 0);

  // clear the first character which means that we don't contain an error
  hostErrorPtr[0] = '\0';

  ErrorMessageBuffer errorMessage(deviceErrorPtr, errorArraySize);

  functor.SetErrorMessageBuffer(errorMessage);

  const std::int64_t blockSizeAsId = 128;
  const std::uint32_t blockSize = 128;
  const std::uint32_t totalBlocks = static_cast<std::uint32_t>(
      (numInstances + blockSizeAsId - 1) / blockSizeAsId);

  Schedule1DIndexKernel<
      Functor><<<totalBlocks, blockSize, 0, cudaStreamPerThread>>>(
      functor, std::int64_t(0), numInstances);

  // sync so that we can check the results of the call.
  // In the future I want move this before the schedule call, and throwing
  // an exception if the previous schedule wrote an error. This would help
  // cuda to run longer before we hard sync.
  cudaStreamSynchronize(cudaStreamPerThread);

  // check what the value is
  if (hostErrorPtr[0] != '\0') {
    std::cerr << hostErrorPtr << std::endl;
  }
}

//-----------------------------------------------------------------------------
template <typename T>
struct FieldTests : public FunctorBase
{
  __device__
  void operator()(std::size_t) const
  {
    T x = 0;
    SignBit(x) == 0;
  }
};

//-----------------------------------------------------------------------------
struct ScalarFieldTests
{
  template <typename T>
  void operator()(const T&) const
  {
    Schedule(FieldTests<T>(), 1);
  }
};


//-----------------------------------------------------------------------------
int main(int, char* [])
{
  using scalars = list<float,double>;

  std::cout << "Tests for scalar types." << std::endl;
  ScalarFieldTests sft;

  //ForEach is the problem
  ListForEach(sft, scalars());

  return 0;
}
