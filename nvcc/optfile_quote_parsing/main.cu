


#ifdef _MSC_VER
#  pragma pack(push, 1)
#  undef PACKED_DEFINE
#  define PACKED_DEFINE
#endif
struct PACKED_DEFINE result_type
{
  bool valid;
  int value;
};

USING_RT = result_type;

#ifdef _MSC_VER
#  pragma pack(pop)
#endif

int main() { return 0; }
