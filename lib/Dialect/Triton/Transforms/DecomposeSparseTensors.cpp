#include <stack>

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

struct DecomposedInfo {
private:
  Value sparseVals;
  Value sparseMeta;
public:
  DecomposedInfo() = default;
  DecomposedInfo(Value sparseVals, Value sparseMeta)
      : sparseVals(sparseVals), sparseMeta(sparseMeta) {}

  Value getValues() { return sparseVals; }
  Value getMeta() { return sparseMeta; }

  std::vector<Value> getValuesShape(OpBuilder &builder,
                                    const Location &loc,
                                    OperandRange shape,
                                    Attribute sparseEncoding) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(isa<triton::SparseAttr>(sparseEncoding));

    int64_t dim2 = 0; // The final values for the shape

    Value dim2Val;

    // Ensure the OperandRange has at least two elements
    assert(shape.size() >= 2 && "OperandRange must have at least two elements");

    // Ignoring first dimension for now assuming 2:4 TODO : (Arya) Fix this later
    // Compute the first dimension of the shape

    // Compute the second dimension of the shape
    if (auto *defOp = shape[1].getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          dim2 = intAttr.getInt();
          dim2 /= 2; // 2:4 has half the columns
          dim2Val = Value(builder.create<arith::ConstantIntOp>(loc, dim2, builder.getI64Type()));
        }
      } else {
        // if the dimension is not a constant, then it has been dynamically computed.
        // We have to divide the value by 2 dynamically.
        dim2Val = Value(shape[1]);
        auto shiftVal = Value(builder.create<arith::ConstantIntOp>(loc, 1, dim2Val.getType()));
        dim2Val = Value(builder.create<arith::ShRSIOp>(loc, dim2Val, shiftVal));
      }
    }
    std::vector<Value> newShape {shape[0], dim2Val};
    return newShape;
  }

  std::vector<Value> getValuesStrides(OpBuilder &builder,
                                      const Location &loc,
                                      OperandRange strides,
                                      Attribute sparseEncoding) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(isa<triton::SparseAttr>(sparseEncoding));

    int64_t dim1 = 0;

    Value dim1Val;

    // Ensure the OperandRange has at least two elements
    assert(strides.size() >= 2 && "OperandRange must have at least two elements");

    // Skipping second dimension because assuming 2:4 TODO : (Arya) Fix this later
    // Compute the first dimension of the shape
    if (auto *defOp = strides[0].getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          dim1 = intAttr.getInt();
          dim1 /= 2;
          dim1Val = Value(builder.create<arith::ConstantIntOp>(loc, dim1, builder.getI64Type()));
        }
      } else {
        // if the dimension is not a constant, then it has been dynamically computed.
        // We have to divide the value by 2 dynamically.
        dim1Val = Value(strides[0]);
        auto shiftVal = Value(builder.create<arith::ConstantIntOp>(loc, 1, dim1Val.getType()));
        dim1Val = Value(builder.create<arith::ShRSIOp>(loc, dim1Val, shiftVal));
      }
    }

    std::vector<Value> newStrides {dim1Val, strides[1]};
    return newStrides;
  }

  std::vector<Value> getValuesOffsets(OpBuilder &builder,
                                      const Location &loc,
                                      OperandRange offsets,
                                      Attribute sparseEncoding) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(isa<triton::SparseAttr>(sparseEncoding));

    int32_t dim2 = 0; // The final values for the shape

    Value dim2Val;

    // Ensure the OperandRange has at least two elements
    assert(offsets.size() >= 2 && "OperandRange must have at least two elements");

    // Compute the first dimension of the shape
    // TODO : (Arya) Commenting out for now because nothing happens in the rows dimension
    // we need to do something different for non - 2:4 sparsity formats

    // Compute the second dimension of the shape
    if (auto *defOp = offsets[1].getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          dim2 = intAttr.getInt();
          dim2 /= 2;
          dim2Val = Value(builder.create<arith::ConstantIntOp>(loc, dim2, builder.getI32Type()));
        }
      } else {
        // if the dimension is not a constant, then it has been dynamically computed.
        // We have to divide the value by 2 dynamically.
        dim2Val = Value(offsets[1]);
        auto shiftVal = Value(builder.create<arith::ConstantIntOp>(loc, 1, dim2Val.getType()));
        dim2Val = Value(builder.create<arith::ShRSIOp>(loc, dim2Val, shiftVal));
      }
    }

    std::vector<Value> newOffsets {offsets[0], dim2Val};
    return newOffsets;
  }

  std::vector<int32_t> getValuesTensorShape(RankedTensorType tensor) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(tensor.getEncoding() && isa<triton::SparseAttr>(tensor.getEncoding()));

    auto tensorShape = tensor.getShape();
    std::vector<int32_t> newTensorShape {static_cast<int32_t>(tensorShape[0]),
                                        static_cast<int32_t>(tensorShape[1]/2)};

    return newTensorShape;
  }

  std::vector<Value> getMetaShape(OpBuilder &builder,
                                  const Location &loc,
                                  OperandRange shape,
                                  Attribute sparseEncoding) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(isa<triton::SparseAttr>(sparseEncoding));

    int64_t dim1 = 0;
    int64_t dim2 = 0; // The final values for the shape

    Value dim1Val;
    Value dim2Val;

    // Ensure the OperandRange has at least two elements
    assert(shape.size() >= 2 && "OperandRange must have at least two elements");

    // Skipping first dimension because assuming 2:4 TODO : (Arya) Fix this later
    // Compute the second dimension of the shape
    if (auto *defOp = shape[0].getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          dim1 = intAttr.getInt();
          assert(dim1 % 16 == 0 && "The number of rows must be divisible by 16");
          dim1 /= 16; // 2:4 metadata has 1/16 of the rows
          dim1Val = Value(builder.create<arith::ConstantIntOp>(loc, dim1, builder.getI64Type()));
        }
      } else {
        // if the dimension is not a constant, then it has been dynamically computed.
        // We have to divide the value by 16 dynamically.
        dim1Val = Value(shape[0]);
        auto shiftVal = Value(builder.create<arith::ConstantIntOp>(loc, 4, dim1Val.getType()));
        dim1Val = Value(builder.create<arith::ShRSIOp>(loc, dim1Val, shiftVal));
      }
    }

    std::vector<Value> newShape {dim1Val, shape[1]};
    return newShape;
  }

  std::vector<Value> getMetaStrides(OpBuilder &builder,
                                    const Location &loc,
                                    OperandRange strides,
                                    Attribute sparseEncoding) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(isa<triton::SparseAttr>(sparseEncoding));

    int64_t dim1 = 0;

    Value dim1Val;

    // Ensure the OperandRange has at least two elements
    assert(strides.size() >= 2 && "OperandRange must have at least two elements");

    std::vector<Value> newStrides {strides[0], strides[1]};
    return newStrides;
  }

  std::vector<Value> getMetaOffsets(OpBuilder &builder,
                                    const Location &loc,
                                    OperandRange offsets,
                                    Attribute sparseEncoding) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(isa<triton::SparseAttr>(sparseEncoding));

    int32_t dim1 = 0;
    int32_t dim2 = 0; // The final values for the shape

    Value dim1Val;
    Value dim2Val;

    // Ensure the OperandRange has at least two elements
    assert(offsets.size() >= 2 && "OperandRange must have at least two elements");

    // Compute the first dimension of the shape
    // TODO : (Arya) Commenting out for now because nothing happens in the rows dimension
    // we need to do something different for non - 2:4 sparsity formats
    if (auto *defOp = offsets[0].getDefiningOp()) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
        if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
          dim1 = intAttr.getInt();
          dim1 /= 16;
          dim1Val = Value(builder.create<arith::ConstantIntOp>(loc, dim1, builder.getI32Type()));
        }
      } else {
        // if the dimension is not a constant, then it has been dynamically computed.
        // We have to divide the value by 16 dynamically.
        dim1Val = Value(offsets[0]);
        auto shiftVal = Value(builder.create<arith::ConstantIntOp>(loc, 4, dim1Val.getType()));
        dim1Val = Value(builder.create<arith::ShRSIOp>(loc, dim1Val, shiftVal));
      }
    }

    std::vector<Value> newOffsets {dim1Val, offsets[1]};
    return newOffsets;
  }

  std::vector<int32_t> getMetaTensorShape(RankedTensorType tensor) {
    // TODO : (Arya) Right now this logic is only for 2:4 matrices
    assert(tensor.getEncoding() && isa<triton::SparseAttr>(tensor.getEncoding()));

    auto tensorShape = tensor.getShape();
    std::vector<int32_t> newTensorShape {static_cast<int32_t>(tensorShape[0]/16),
                                        static_cast<int32_t>(tensorShape[1])};

    return newTensorShape;
  }

};

class DecomposeSparseTensorsPass
    : public TritonDecomposeSparseTensorsBase<DecomposeSparseTensorsPass> {

private:
  DenseMap<Value, DecomposedInfo> decomposedInfo;
  llvm::SmallVector<Attribute> argAttrs;

public:
  bool isATensorPtr(Type type) {
    if (auto ptrTy = dyn_cast<triton::PointerType>(type)) {
      return isa<RankedTensorType>(ptrTy.getPointeeType());
    }
    return false;
  }

  Operation *processOperands(Operation *op) {
    // TODO : (Arya) Think about when we would want to replace a value with its metadata
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto currentOperand = op->getOperand(i);
      if (decomposedInfo.count(currentOperand)) {
        // Replace the operand with the values' decomposed tensor
        op->setOperand(i, decomposedInfo[currentOperand].getValues());
      }
    }
    return op;
  }

  Operation *decomposeForOp(OpBuilder &builder,
                            scf::ForOp op,
                            std::stack<Operation *> &eraser) {
    // Decompose the initial arguments to the scf For loop
    // These decomposed arguments are used to create a new for loop
    // that will replace the current one.
    auto initArgs = op.getInitArgs();
    std::vector<Value> newInitArgs;
    for (Value v : initArgs) {
      if (decomposedInfo.count(v) > 0) {
        auto info = decomposedInfo[v];
        newInitArgs.push_back(info.getValues());
        newInitArgs.push_back(info.getMeta());
      } else
        newInitArgs.push_back(v);
    }

    // Create the new ForOp with the new Init Args
    auto newForOp = builder.create<scf::ForOp>(op.getLoc(),
                                               op.getLowerBound(),
                                               op.getUpperBound(),
                                               op.getStep(),
                                               newInitArgs);

    // Decompose the iter args as well. This will help create a mapping
    // from the old for loop to the new one; ensuring we can replace
    // sparse ops with their dense counterparts when decomposition is called
    // on the body.
    auto iterArgs = op.getRegionIterArgs();
    assert(initArgs.size() == iterArgs.size());

    for (int i = 0, oldI = 0, sz = iterArgs.size(); oldI < sz; i++, oldI++) {
      if (decomposedInfo.count(initArgs[oldI]) > 0) {
        // if this initial value of this iter arg has been decomposed, then it is sparse
        // Therefore, we need to store its decomposition so that it is later
        // used to decompose values in the body of the for loop
        decomposedInfo[iterArgs[oldI]] = DecomposedInfo(newForOp.getRegionIterArg(i),
                                                        newForOp.getRegionIterArg(i+1));
        i++;
      }
    }


    // Create mappings from the old iter arguments to the new ones.
    // There are two cases:
    // 1. The old argument is a sparse tensor. In this case, we can have
    //    an identity mapping for the tensor. Although it will relate to
    //    arguments from the old ForOp, these arguments will later get replaced
    //    in the decomposition pipeline
    // 2. In the case that the argument is not a sparse tensor, we must
    //    correctly map the argument from the old ForOp to its equivalent
    //    argument created in the new ForOp.
    IRMapping argMap;
    for (int i = 0, oldI = 0, sz = initArgs.size();
         oldI < sz;
         i++, oldI++) {
      auto oldRegionArg = iterArgs[oldI];

      auto ptrType = dyn_cast<triton::PointerType>(oldRegionArg.getType());
      if (ptrType && triton::isSparseTensorPointer(ptrType)) {
        argMap.map(oldRegionArg, oldRegionArg);
        // TODO : (Arya) If the sparse tensor has more than 1
        // metadata value, then we have to increase i by more than one
        i++;
      } else {
        argMap.map(oldRegionArg, newForOp.getRegionIterArg(i));
      }
    }
    argMap.map(op.getInductionVar(), newForOp.getInductionVar());

    // Clone the body of the old ForOp to the new ForOp
    builder.setInsertionPointToStart(newForOp.getBody());
    for (auto &opInFor : *op.getBody()) {
      auto *newOp = builder.clone(opInFor, argMap);
      for (unsigned i = 0; i < opInFor.getNumResults(); ++i)
        argMap.map(op->getResult(i), newOp->getResult(i));
    }

    // Replace the old results with the new results
    assert(op.getNumResults() == op.getInitArgs().size());
    for (unsigned i = 0, oldI = 0; i < newForOp.getNumResults(); ++i, ++oldI) {
      auto oldResult = op.getResult(oldI);
      triton::PointerType ptrType = dyn_cast<triton::PointerType>(oldResult.getType());
      if (ptrType && triton::isSparseTensorPointer(ptrType)) {
        decomposedInfo[oldResult] = DecomposedInfo(newForOp.getResult(i), newForOp.getResult(i + 1));
        ++i; // TODO(victor): Change if decomposes into more elements
      } else {
        oldResult.replaceAllUsesWith(newForOp.getResult(i));
      }
    }

    // Erase the old ForOp later
    eraser.push(op);

    // return the new ForOp to be decomposed by the class.
    return newForOp;
  }

  Operation *decomposeYieldOp(OpBuilder &builder,
                              scf::YieldOp op,
                              std::stack<Operation *> &eraser) {
    std::vector<Value> newOperands;
    for (unsigned i = 0, sz = op.getNumOperands(); i < sz; i++) {
      auto operand = op->getOperand(i);
      triton::PointerType ptrType = dyn_cast<triton::PointerType>(operand.getType());
      if (ptrType && triton::isSparseTensorPointer(ptrType)) {
        assert(decomposedInfo.count(operand) > 0);
        auto info = decomposedInfo[operand];
        newOperands.push_back(info.getValues());
        newOperands.push_back(info.getMeta());
      } else if (decomposedInfo.count(operand) > 0) {
        // If the operand is not a sparse pointer but it has been "decomposed",
        // then it must be replaced by its decomposed value.
        newOperands.push_back(decomposedInfo[operand].getValues());
      } else
        newOperands.push_back(operand);
    }

    op->setOperands(newOperands);

    return nullptr;
  }

  Operation *decomposeAdvanceOp(OpBuilder &builder,
                                triton::AdvanceOp op,
                                std::stack<Operation *> &eraser) {
    // TODO : (Arya) Advance op should behave differently for different sparsity types.
    //  Assuming 2:4 sparsity right now
    auto ptr = op.getPtr();
    auto offsets = op.getOffsets();

    // Only decompose this op if it's advancing on a sparse pointer
    if (!triton::isSparseTensorPointer(ptr.getType())) return op;

    assert(decomposedInfo.count(ptr));
    assert(offsets.size() == 2);
    auto info = decomposedInfo[ptr];

    // compute new offsets for the values and metadata
    std::vector<Value> valueOffsets(offsets.size());
    std::vector<Value> metaOffsets(offsets.size());
    for (int dim = 0; dim < offsets.size(); dim++) {
      if (dim != 1)
        valueOffsets[dim] = offsets[dim];
      else
        valueOffsets[dim] = builder.create<arith::DivSIOp>(op.getLoc(),
                                                           offsets[dim],
                                                           Value(builder.create<arith::ConstantIntOp>(
                                                               op.getLoc(), 2, builder.getI32Type()
                                                                   )));
    }
    for (int dim = 0; dim < offsets.size(); dim++) {
      if (dim == 0)
        metaOffsets[dim] = builder.create<arith::DivSIOp>(op.getLoc(),
                                                          offsets[dim],
                                                          Value(builder.create<arith::ConstantIntOp>(
                                                              op.getLoc(), 16, builder.getI32Type()
                                                                  )));
      else
        metaOffsets[dim] = offsets[dim];
    }
    auto valsOp = builder.create<triton::AdvanceOp>(op.getLoc(),
                                                    info.getValues().getType(),
                                                    info.getValues(),
                                                    valueOffsets);
    auto metaOp = builder.create<triton::AdvanceOp>(op.getLoc(),
                                                    info.getMeta().getType(),
                                                    info.getMeta(),
                                                    metaOffsets);

    decomposedInfo[op] = DecomposedInfo(valsOp, metaOp);
    eraser.push(op);

    return nullptr;
  }

  Operation *decomposeDotOp(OpBuilder &builder,
                            triton::DotOp op,
                            std::stack<Operation *> &eraser) {
    assert(!op.getE()); // Metadata should not already exist

    bool aSparse;
    triton::DotOp result;

    auto lhs = op.getA();
    auto rhs = op.getB();

    if(!lhs.getType().getEncoding() &&
        !rhs.getType().getEncoding())
      // The dot operation is not concerned with sparsity
      return op;

    if (lhs.getType().getEncoding() &&
        isa<triton::SparseAttr>(lhs.getType().getEncoding())) {
      aSparse = true;
    } else {
      assert(isa<triton::SparseAttr>(rhs.getType().getEncoding()));
      aSparse = false;
    }

    if (aSparse) {
      assert(decomposedInfo.count(lhs));
      auto info = decomposedInfo[lhs];
      result = builder.create<triton::DotOp>(
          op.getLoc(),
          info.getValues(),
          op.getB(), op.getC(),
          info.getMeta(),
          op.getInputPrecision(), op.getMaxNumImpreciseAcc(),
          /*sparseIndex*/0);
    } else {
      assert(decomposedInfo.count(rhs));
      auto info = decomposedInfo[rhs];
      result = builder.create<triton::DotOp>(
          op.getLoc(),
          op.getA(),
          info.getValues(), op.getC(),
          info.getMeta(),
          op.getInputPrecision(), op.getMaxNumImpreciseAcc(),
          /*sparseIndex*/1);
    }

    decomposedInfo[op] = DecomposedInfo(result, nullptr);
    eraser.push(op);

    return nullptr;
  }

  Operation *decomposeSparseValsOp(OpBuilder &builder,
                                   triton::SparseValuesOp op,
                                   std::stack<Operation *> &eraser) {
    assert(decomposedInfo.count(op.getBase()));

    auto info = decomposedInfo[op.getBase()];
    decomposedInfo[op.getResult()] = DecomposedInfo(info.getValues(), nullptr);

    eraser.push(op);

    return nullptr;
  }

  Operation *decomposeLoadOp(OpBuilder &builder, triton::LoadOp op,
                             std::stack<Operation *> &eraser) {
    if(decomposedInfo.count(op.getPtr()) == 0) return op;

    assert(isATensorPtr(op.getPtr().getType()));
    assert(decomposedInfo.count(op.getPtr()));

    auto info = decomposedInfo[op.getPtr()];

    auto valsOp = builder.create<triton::LoadOp>(op.getLoc(), info.getValues(),
                                                 op.getBoundaryCheck(), op.getPadding(),
                                                 op.getCache(), op.getEvict(),
                                                 op.getIsVolatile());

    auto metaOp = builder.create<triton::LoadOp>(op.getLoc(), info.getMeta(),
                                                 op.getBoundaryCheck(), op.getPadding(),
                                                 op.getCache(), op.getEvict(),
                                                 op.getIsVolatile());

    decomposedInfo[op] = DecomposedInfo(valsOp, metaOp);
    eraser.push(op);

    return nullptr;
  }

  Operation *decomposeMakeTensorPtr(OpBuilder &builder,
                                    triton::MakeTensorPtrOp op,
                                    std::stack<Operation *> &eraser) {
    if (!op.getBase().getType().getEncoding() ||
        !isa<triton::SparseAttr>(op.getBase().getType().getEncoding())) return op;
    assert(decomposedInfo.count(op.getBase()));
    auto info = decomposedInfo[op.getBase()];

    auto shape = op.getShape();
    auto strides = op.getStrides();
    auto offsets = op.getOffsets();

    std::vector<Value> valuesShape, metaShape, valuesStrides, metaStrides,
        valuesOffsets, metaOffsets;

    std::vector<int32_t> valuesTensorShape, metaTensorShape, valuesOrder,
        metaOrder;

    if (auto ptrTy = dyn_cast<triton::PointerType>(op.getResult().getType())) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(ptrTy.getPointeeType())) {
        assert(tensorTy.getEncoding() &&
               isa<triton::SparseAttr>(tensorTy.getEncoding()));
        auto sparseEncoding = cast<triton::SparseAttr>(tensorTy.getEncoding());
        valuesTensorShape = info.getValuesTensorShape(tensorTy);
        valuesShape =
            info.getValuesShape(builder, op.getLoc(), shape, sparseEncoding);
        valuesStrides = info.getValuesStrides(builder, op.getLoc(), strides,
                                              sparseEncoding);
        valuesOffsets = info.getValuesOffsets(builder, op.getLoc(), offsets,
                                              sparseEncoding);
        auto valsOp = builder.create<triton::MakeTensorPtrOp>(
            op.getLoc(), info.getValues(), valuesShape, valuesStrides,
            valuesOffsets, valuesTensorShape, op.getOrder());

        metaTensorShape = info.getMetaTensorShape(tensorTy);
        metaShape =
            info.getMetaShape(builder, op.getLoc(), shape, sparseEncoding);
        metaStrides =
            info.getMetaStrides(builder, op.getLoc(), strides, sparseEncoding);
        metaOffsets =
            info.getMetaOffsets(builder, op.getLoc(), offsets, sparseEncoding);
        auto metaOp = builder.create<triton::MakeTensorPtrOp>(
            op.getLoc(), info.getMeta(), metaShape, metaStrides, metaOffsets,
            metaTensorShape, op.getOrder());

        decomposedInfo[op] = DecomposedInfo(valsOp, metaOp);
        eraser.push(op);
      }
    }

    return nullptr;
  }

  Operation *decomposeFunctionArgs(OpBuilder &builder, triton::FuncOp op,
                                   std::stack<Operation *> &eraser) {
    // Set the arg attributes to the old arg attributes
    auto oldArgAttrs = op.getAllArgAttrs();
    llvm::SmallVector<Attribute> newArgAttrs(oldArgAttrs.getValue());
    argAttrs = newArgAttrs;
    for (int i = 0; i < op.getNumArguments(); i++) {
      auto numDecompArgs = 0; // number of new args the args to which this arg will be decomposed
      auto arg = op.getArgument(i);
      auto pointerTy = dyn_cast<triton::PointerType>(arg.getType());
      if (pointerTy && triton::isSparsePointer(pointerTy)) {
        // Generate new Arguments
        auto argValsType =
            triton::PointerType::get(pointerTy.getPointeeType(), 1);
        auto argMetaElType =
            triton::getSparsityMetadataElementType(pointerTy.getEncoding());
        auto argMetaType = triton::PointerType::get(argMetaElType, 1);

        // Modify function type for the new Arguments
        auto funcTy = op.getFunctionType();
        auto newInputTy = llvm::to_vector<4>(funcTy.getInputs());

        // Add the new arguments in
        newInputTy.insert(newInputTy.begin() + i, argMetaType); numDecompArgs++;
        newInputTy.insert(newInputTy.begin() + i, argValsType); numDecompArgs++;
        auto newFuncTy = FunctionType::get(funcTy.getContext(), newInputTy,
                                           funcTy.getResults());

        // Add the new arguments to the region
        op.setType(newFuncTy);

        Block &entryBlock = op.getBody().front();
        auto argMeta =
            entryBlock.insertArgument(i, argMetaType, arg.getLoc());
        auto argVals =
            entryBlock.insertArgument(i, argValsType, arg.getLoc());

        // Move the arg attributes accordingly
        // The new attributes cannot be added until the argument is erased. We save the new attributes
        // for the end of the pass
        newArgAttrs.insert(newArgAttrs.begin()+i+1, *(newArgAttrs.begin()+i));
        argAttrs = newArgAttrs;

        decomposedInfo[arg] = DecomposedInfo(argVals, argMeta);

        i += numDecompArgs;
      }
    }
    return op;
  }

  Operation *decomposeOp(Operation *op, std::stack<Operation *> &eraser) {
    OpBuilder builder(op);
    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      return decomposeFunctionArgs(builder, funcOp, eraser);
    } else if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
      return decomposeMakeTensorPtr(builder, makeTensorPtrOp, eraser);
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      return decomposeLoadOp(builder, loadOp, eraser);
    } else if (auto sparseValsOp = dyn_cast<triton::SparseValuesOp>(op)) {
      return decomposeSparseValsOp(builder, sparseValsOp, eraser);
    } else if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      return decomposeDotOp(builder, dotOp, eraser);
    } else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
      return decomposeAdvanceOp(builder, advanceOp, eraser);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return decomposeForOp(builder, forOp, eraser);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      return decomposeYieldOp(builder, yieldOp, eraser);
    } else {
      // The dot operation is the only place we  use the metadata.
      // If a sparse operand is being used in any other place, we replace
      // its value with the values array of the sparse tensor
      processOperands(op);
    }
    return op;
  }

  void eraseSparseFunctionArgs(triton::FuncOp *funcOp) {
    for (auto arg : funcOp->getArguments()) {
      auto pointerTy = dyn_cast<triton::PointerType>(arg.getType());
      if (pointerTy && triton::isSparsePointer(pointerTy)) {
        auto index = arg.getArgNumber();

        // Modify function type for the new Arguments
        auto funcTy = funcOp->getFunctionType();
        auto newInputTy = llvm::to_vector<4>(funcTy.getInputs());
        // erase the argument type
        newInputTy.erase(newInputTy.begin() + index);
        auto newFuncTy = FunctionType::get(funcTy.getContext(), newInputTy,
                                           funcTy.getResults());

        // Set the new function Type and remove the argument from the block
        funcOp->setType(newFuncTy);

        Block &entryBlock = funcOp->getBody().front();
        entryBlock.eraseArgument(index);
      }
    }
    funcOp->setAllArgAttrs(argAttrs);
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        // Another copy to not break iterator after erasing operations
        SmallVector<Operation *> blockCopy;
        for (auto &nestedOp : block)
          blockCopy.push_back(&nestedOp);

        for (auto &nestedOp : blockCopy) {
          if (auto newOp = decomposeOp(nestedOp, eraser)) {
            visitOperation(newOp, eraser);
          }
        }
      }
    }
  }

  void runOnOperation() override {
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser);

    decomposedInfo.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }

    // After decomposing every sparse argument in every function,
    // we have to delete the sparse arguments. This was initially not done
    // because we cannot delete an argument whose value is used in the function. This is now possible since the operations using those arguments have been erased.
    for (auto &region : getOperation()->getRegions()) {
      for (auto &block : region) {
        for (auto &nestedOp : block) {
          if (auto funcOp = dyn_cast<triton::FuncOp>(nestedOp)) {
            eraseSparseFunctionArgs(&funcOp);
          }
        }
      }
    }
  }
};

std::unique_ptr<Pass> triton::createDecomposeSparseTensorsPass() {
  return std::make_unique<DecomposeSparseTensorsPass>();
}