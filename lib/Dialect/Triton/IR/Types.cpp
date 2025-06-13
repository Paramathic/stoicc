#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void TritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Triton/IR/Types.cpp.inc"
      >();
}

Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  int addressSpace = 1;
  Attribute encoding;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(addressSpace))
      return Type();
    if (parser.parseOptionalInteger(addressSpace).has_value()) {
      // Check if there's another comma for encoding
      if (parser.parseOptionalComma()) {
        if (parser.parseAttribute(encoding)) {
          parser.emitError(parser.getCurrentLocation(),
                           "expected encoding after address space");
          return {};
        }
      }
    } else {
      // If not an integer, assume it's the encoding
      if (parser.parseAttribute(encoding))
        return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return PointerType::get(pointeeType, addressSpace, encoding);
}

void PointerType::print(AsmPrinter &printer) const {
  printer << "<" << getPointeeType();
  if (getAddressSpace() != 1) {
    printer << ", " << getAddressSpace();
  } if (getEncoding()) {
    printer << ", " << getEncoding();
  }
  printer << ">";
}

namespace mlir {

namespace triton {

unsigned getPointeeBitWidth(Type type) {
  auto pointeeType = getPointeeType(type);
  if (auto tensorTy = dyn_cast<RankedTensorType>(pointeeType))
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  return pointeeType.getIntOrFloatBitWidth();
}

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorTy.getShape(), i1Type,
                                 tensorTy.getEncoding());
  return i1Type;
}

Type getPointeeType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // Tensor of pointers
    auto shape = tensorTy.getShape();
    auto ptrType = dyn_cast<PointerType>(tensorTy.getElementType());
    Type pointeeType = ptrType.getPointeeType();
    return RankedTensorType::get(shape, pointeeType, tensorTy.getEncoding());
  } else if (auto ptrType = dyn_cast<PointerType>(type)) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return type;
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return RankedTensorType::get(tensorTy.getShape(), i32Type,
                                 tensorTy.getEncoding());
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorTy.getElementType();
    auto shape = tensorTy.getShape();
    PointerType ptrType = PointerType::get(elementType, 1);
    return RankedTensorType::get(shape, ptrType, tensorTy.getEncoding());
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerTypeToElement(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  PointerType ptrType = PointerType::get(elementType, 1);
  return ptrType;
}

// upstream Triton only uses address space 1 for Pointer Type
Type getPointerType(Type type, int addressSpace) {
  return PointerType::get(type, addressSpace);
}

int getAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return ptrType.getAddressSpace();
  return 1;
}

bool isTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return isa<RankedTensorType>(ptrType.getPointeeType());
  return false;
}

bool isTensorOrTensorPointerType(Type type) {
  return isa<RankedTensorType>(type) || isTensorPointerType(type);
}

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
      return tensorTy.getElementType();
  return {};
}

Type getMaybeSparseTensorPointerType(Type type, int addresSpace) {
  auto elementType = getElementTypeOfTensorPointerType(type);
  if (auto ptrType = dyn_cast<PointerType>(type)) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      if (auto encoding = tensorTy.getEncoding())
        return PointerType::get(elementType, addresSpace, encoding);
      else
        return PointerType::get(elementType, addresSpace);
    }
  }
  return {};
}

bool isSparseTensor(RankedTensorType type) {
  if (auto encoding = type.getEncoding()) {
    return dyn_cast<SparseAttr>(type.getEncoding()) != NULL;
  }
  return false;
}

bool isSparseTensorPointer(PointerType type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type.getPointeeType())) {
    return isSparseTensor(tensorType);
  }
  return false;
}

bool isSparsePointer(PointerType type) {
  if (auto encoding = type.getEncoding()) {
    return isa<triton::SparseAttr>(encoding);
  }
  return false;
}

SmallVector<int64_t> getDenseTensorValuesShape(RankedTensorType type) {
  // TODO : (Arya) Right now, this returns the type for a 2:4 tensor.
  //  Change it to handle more sparsity types
  /* Given a sparse tensor type, return the shape of the values tensor
   * If tensor is dense, return its original shape */
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    if (isSparseTensor(tensorTy)){
      auto tensorShape = tensorTy.getShape();
      SmallVector<int64_t> newShape {tensorShape[0], tensorShape[1]/2};
      return newShape;
    }
    auto shape = tensorTy.getShape();
    return SmallVector<int64_t>(shape.begin(), shape.end());
  }
  return {};
}

Type getSparsityMetadataElementType(Attribute encoding) {
  if (auto sparseEncoding = dyn_cast<triton::SparseAttr>(encoding)) {
    // TODO : (Arya) Return different metadata types for different encodings
    return IntegerType::get(encoding.getContext(), 16);
  }
  return {};
}

// TODO : (Arya) Write logic for checking that the encoding is 2:4
Type getMaybeSparsePointeeType(Type type) {
  /** The type is either a Tensor of pointers or a pointer to a tensor.
  The tensor may be either sparse or dense. Either way, a dense tensor type
  should be returned.
  e.g. fn(tensor<128x128x!tt.ptr<f16>, #sparse<"NV24">>) -> tensor<128x64xf16>
       fn(tensor<128x128x!tt.ptr<f16>>) -> tensor<128x128xf16>
       fn(!tt.ptr<tensor<128x128xf16, #sparse<"NV24">>) -> tensor<128x64xf16>
       fn(!tt.ptr<tensor<128x128xf16>>) -> tensor<128x128xf16>
       */
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // Tensor of pointers
    auto shape = getDenseTensorValuesShape(tensorTy);
    auto ptrType = dyn_cast<PointerType>(tensorTy.getElementType());
    Type pointeeType = ptrType.getPointeeType();
    if (isSparseTensor(tensorTy)) {
      return RankedTensorType::get(shape, pointeeType);
    }
    return RankedTensorType::get(shape, pointeeType, tensorTy.getEncoding());
  } else if (auto ptrType = dyn_cast<PointerType>(type)) {
    // scalar pointer
    auto elementType = getElementTypeOfTensorPointerType(type);
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      if (isSparseTensor(tensorTy)) {
        auto newShape = getDenseTensorValuesShape(tensorTy);
        return RankedTensorType::get(newShape, tensorTy.getElementType());
      } else
        return tensorTy;
    }
    return ptrType.getPointeeType();
  }
  return type;
}

} // namespace triton

} // namespace mlir
