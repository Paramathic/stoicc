#include <memory>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace mlir::triton {
namespace {

bool isZero(Value val) {
  if (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()))
    return true;
  // broadcast(constant_0)
  if (auto bc = val.getDefiningOp<BroadcastOp>()) {
    if (matchPattern(bc.getSrc(), m_Zero()) ||
        matchPattern(bc.getSrc(), m_AnyZeroFloat()))
      return true;
  }
  return false;
}

bool isBroadcastConstantCombinable(Attribute value) {
  if (auto denseValue = dyn_cast<DenseElementsAttr>(value)) {
    return denseValue.isSplat();
  }
  return isa<FloatAttr, IntegerAttr>(value);
}

DenseElementsAttr getConstantValue(Builder &builder, Attribute value,
                                   Value bcast_res) {
  auto resType = cast<ShapedType>(bcast_res.getType());
  DenseElementsAttr res;
  if (auto denseValue = dyn_cast<DenseElementsAttr>(value)) {
    res =
        DenseElementsAttr::get(resType, denseValue.getSplatValue<Attribute>());
  } else {
    res = DenseElementsAttr::get(resType, value);
  }
  return res;
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto *loadOpCandidate = trueValue.getDefiningOp();
    auto loadOp = llvm::dyn_cast_or_null<LoadOp>(loadOpCandidate);
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto *splatOpCandidate = mask.getDefiningOp();
    auto splatOp = llvm::dyn_cast_or_null<SplatOp>(splatOpCandidate);
    if (!splatOp)
      return failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().getIntOrFloatBitWidth() <= 32;
    return false;
  }

  static SmallVector<int> getEqualIndices(ArrayRef<int64_t> x,
                                          ArrayRef<int64_t> y) {
    SmallVector<int> res;
    for (int i = 0; i < x.size(); ++i)
      if (x[i] == y[i])
        res.push_back(i);
    return res;
  }

public:
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = llvm::dyn_cast_or_null<arith::MulFOp>(
        reduceOp.getOperand(0).getDefiningOp());
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = llvm::dyn_cast_or_null<BroadcastOp>(
        mulOp.getOperand(0).getDefiningOp());
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = llvm::dyn_cast_or_null<BroadcastOp>(
        mulOp.getOperand(1).getDefiningOp());
    if (!broadcastRhsOp)
      return failure();
    // broadcast operand is expand dims
    auto expandLhsOp = llvm::dyn_cast_or_null<ExpandDimsOp>(
        broadcastLhsOp.getSrc().getDefiningOp());
    if (!expandLhsOp)
      return failure();
    auto expandRhsOp = llvm::dyn_cast_or_null<ExpandDimsOp>(
        broadcastRhsOp.getSrc().getDefiningOp());
    if (!expandRhsOp)
      return failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]},
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType());
    rewriter.setInsertionPoint(op);
    auto newAcc = rewriter.create<SplatOp>(
        op->getLoc(), newAccType,
        rewriter.create<arith::ConstantOp>(op->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<DotOp>(op, expandLhsOp.getSrc(),
                                       expandRhsOp.getSrc(), newAcc,
                                       InputPrecision::TF32, 0);
    return success();
  }
};

// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
template <typename OpTy>
class CombineDotAddPattern : public RewritePattern {
public:
  // TODO(Victor): Value for benefit?
  CombineDotAddPattern(MLIRContext *context) : RewritePattern(OpTy::getOperationName(), 3, context) { }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto addOp = llvm::dyn_cast<OpTy>(op);
    if (!addOp)
      return failure();

    Value lhs = addOp.getLhs();
    Value rhs = addOp.getRhs();

    auto dotOp = llvm::dyn_cast_or_null<DotOp>(lhs.getDefiningOp());
    Value add;
    if (!dotOp) {
      dotOp = llvm::dyn_cast_or_null<DotOp>(rhs.getDefiningOp());
      if (!dotOp) {
        return failure();
      }
      add = lhs;
    } else {
      add = rhs;
    }

    // Constraints
    if (!isZero(dotOp.getC()) || !dotOp.getResult().hasOneUse())
      return failure();

    auto fpAddOp = llvm::dyn_cast<arith::AddFOp>(op);
    if (fpAddOp && dotOp.getMaxNumImpreciseAcc() != 0)
      return failure();

    rewriter.replaceAllUsesWith(addOp.getResult(), dotOp.getResult());
    rewriter.modifyOpInPlace(dotOp, [&]() {
      dotOp->replaceUsesOfWith(dotOp.getC(), add);
    });
    rewriter.eraseOp(addOp);
    return success();
  }
};

class CombineOpsPass : public TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<CombineDotAddPattern<arith::AddIOp>>(context);
    patterns.add<CombineDotAddPattern<arith::AddFOp>>(context);
    patterns.add<CombineSelectMaskedLoadPattern>(context);
    patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastConstantPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}

} // namespace mlir::triton
