#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.h.inc"

namespace mlir {

namespace triton {

bool isTensorPointerType(Type type);

bool isTensorOrTensorPointerType(Type type);

unsigned getPointeeBitWidth(Type type);

Type getPointeeType(Type type);

Type getPointerType(Type type, int addressSpace = 1);

int getAddressSpace(Type type);

Type getElementTypeOfTensorPointerType(Type type);

Type getMaybeSparseTensorPointerType(Type type, int addressSpace);

Type getMaybeSparsePointeeType(Type type);

Type getI1SameShape(Type type);

Type getI32SameShape(Type type);

Type getPointerTypeSameShape(Type type);

Type getPointerTypeToElement(Type type);

bool isSparseTensor(RankedTensorType type);

bool isSparseTensorPointer(PointerType type);

bool isSparsePointer(PointerType type);

SmallVector<int64_t> getDenseTensorValuesShape(RankedTensorType type);

Type getSparsityMetadataElementType(Attribute encoding);



} // namespace triton

} // namespace mlir

#endif // TRITON_IR_TYPES_H_
