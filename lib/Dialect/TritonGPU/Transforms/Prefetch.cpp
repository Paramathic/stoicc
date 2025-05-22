//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = triton_gpu.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = triton_gpu.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//===----------------------------------------------------------------------===//

#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidth
  unsigned prefetchWidth = 32;
  unsigned metadataPrefetchWidth = 2;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2eLoopArg;
  DenseMap<Value, Value> dot2eHeaderDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  DenseMap<Value, Value> dot2eYield;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  DenseMap<Value, SmallVector<Value>> dot2eVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  LogicalResult isForOpOperand(Value v);

  Value generatePrefetch(Value v, unsigned opIdx, signed meta, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, signed meta, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  SmallVector<int64_t> offset{0, 0};
  Type elementType = type.getElementType();

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? 1 : 0;

  int _prefetchWidth = prefetchWidth;
  if (meta > 0) _prefetchWidth = metadataPrefetchWidth;

  offset[kIdx] = isPrologue ? 0 : _prefetchWidth;
  shape[kIdx] = isPrologue ? _prefetchWidth : (shape[kIdx] - _prefetchWidth);


  if (shapeK) {
    shape[kIdx] = *shapeK;
    if (meta > 0) shape[kIdx] = metadataPrefetchWidth;
  }

  if (offsetK) {
    offset[kIdx] = *offsetK;
    if (meta == 0) offset[kIdx] = *offsetK/2;
    if (meta > 0) offset[kIdx] = metadataPrefetchWidth >= 64? /* If True, we can use ldmatrix */
                                                 *offsetK  :
                                                 *offsetK/(prefetchWidth/metadataPrefetchWidth);
  }

  SmallVector<Value> offsetsVal;
  for (int64_t off : offset)
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), off, 32));
  Value newSmem = builder.create<triton::gpu::MemDescSubviewOp>(
      v.getLoc(),
      triton::MemDescType::get(shape, elementType, type.getEncoding()), v,
      offsetsVal);

  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding,  meta > 0 ? 2 : std::max(2, _prefetchWidth / 8), meta);
  Value prefetchSlice = builder.create<triton::gpu::LocalLoadOp>(
      v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
      newSmem);

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // bail out if there exist non v2 dots.
      auto dstEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      if (!dstEnc || dstEnc.getVersionMajor() != 2)
        return failure();
      dotsInFor.push_back(dotOp);
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    while (op) {
      if (op->getNumOperands() != 1)
        break;
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  auto getYieldOp = [this](Value v) -> Value {
    auto arg = mlir::cast<BlockArgument>(v);
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  for (triton::DotOp dot : dotsInFor) {
    auto dotIsSparse = triton::isSparseDot(dot);

    auto aType = dot.getA().getType();
    auto bType = dot.getB().getType();
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    int aKWidth = aEnc.getKWidth();
    int bKWidth = bEnc.getKWidth();
    assert(aKWidth == bKWidth);

    auto kSize = bType.getShape()[0];

    if (dotIsSparse) {
      auto eType = dot.getE().getType();
      auto eEnc = mlir::cast<triton::gpu::DotOperandEncodingAttr>(eType.getEncoding());
    }

    // works better with nvidia tensor cores
    unsigned elementWidth = aType.getElementTypeBitWidth();
    if (aKWidth == 0)
      prefetchWidth = 256 / elementWidth;
    else
      prefetchWidth = 8 * bKWidth;

    if (dotIsSparse) {
      auto eType = dot.getE().getType();
      auto eEnc = mlir::cast<triton::gpu::DotOperandEncodingAttr>(eType.getEncoding());
      auto eKWidth = eEnc.getKWidth();
      bool canUseLdmatrix = eType.getShape()[eType.getRank()-1] >= 64;
      metadataPrefetchWidth = canUseLdmatrix? 32*eKWidth : prefetchWidth/8;
    }

    // Skip prefetching if kSize is less than prefetchWidth
    if (kSize < prefetchWidth)
      continue;
    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());
    SmallVector<Value> eVals;
    if (dotIsSparse)
      eVals = getPrefetchSrc(dot.getE());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      Value eHeaderDef, eSmem;
      if (eVals.size()) {
        eSmem = eVals.front();
        eHeaderDef = getIncomingOp(eSmem);
      }
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef && (!dotIsSparse || eHeaderDef)) {
        dots.insert(dot);
        dot2aVals[dot] = aVals;
        dot2bVals[dot] = bVals;
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        dot2aYield[dot] = getYieldOp(aSmem);
        dot2bYield[dot] = getYieldOp(bSmem);
        if (eHeaderDef) {
          dot2eVals[dot] = eVals;
          dot2eHeaderDef[dot] = eHeaderDef;
          dot2eLoopArg[dot] = eSmem;
          dot2eYield[dot] = getYieldOp(eSmem);
        }
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (triton::DotOp dot : dots) {
    auto dotIsSparse = triton::isSparseDot(dot);

    Attribute dotEncoding = dot.getType().getEncoding();
    auto aEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getA().getType().getEncoding());
    auto bEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getB().getType().getEncoding());
    assert(aEncoding && bEncoding && "A and B must have DotOperandEncodingAttr types.");

    DotOperandEncodingAttr eEncoding;
    if (dotIsSparse) eEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getE().getType().getEncoding());

    Value aPrefetched =
        generatePrefetch(dot2aHeaderDef[dot], 0, aEncoding.getMeta(), true, dotEncoding, builder);
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched =
        generatePrefetch(dot2bHeaderDef[dot], 1, bEncoding.getMeta(), true, dotEncoding, builder);
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);
    if (dotIsSparse) {
      Value ePrefetched =
          generatePrefetch(dot2eHeaderDef[dot], eEncoding.getOpIdx(), eEncoding.getMeta(), true,
                           dotEncoding, builder);
      cloneElementwiseOps(ePrefetched, dot2eVals[dot], builder);
      operand2headPrefetch[dot.getE()] = ePrefetched;
    }



    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (triton::DotOp dot : dots) {
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
    if (triton::isSparseDot(dot)) loopArgs.push_back(operand2headPrefetch[dot.getE()]);
  }

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  bool canUseLdmatrix = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dot && dots.contains(dot)) {
      auto dotIsSparse = triton::isSparseDot(dot);
      Attribute dotEncoding = dot.getType().getEncoding();
      // prefetched dot
      Operation *firstDot = builder.clone(*dot, mapping);
      Value aArg, eArg;
      if (Value a = operand2headPrefetch.lookup(dot.getA())) {
        aArg = newForOp.getTiedLoopRegionIterArg(&*a.use_begin());
        firstDot->setOperand(0, aArg);
      }
      if (Value b = operand2headPrefetch.lookup(dot.getB()))
        firstDot->setOperand(
            1, newForOp.getTiedLoopRegionIterArg(&*b.use_begin()));

      if (dotIsSparse) {
        if (Value e = operand2headPrefetch.lookup(dot.getE())) {
          eArg = newForOp.getTiedLoopRegionIterArg(&*e.use_begin());
          firstDot->setOperand(
              3, eArg);
          canUseLdmatrix = dot.getE().getType().getShape()[1] >= 64;
          if (canUseLdmatrix) {
            // set the sparsity selector for the first dot to 0
            auto _dot = dyn_cast<triton::DotOp>(firstDot);
            _dot.setSpSelector(0);
          }
        }
      }

      // remaining part
      int64_t kOff = prefetchWidth;
      int64_t kRem = dot.getB().getType().getShape()[0] - prefetchWidth;
      Operation *prevDot = firstDot;
      auto spSel = 1; // Sparsity Selector
      while (kRem != 0) {
        // int64_t kShape = largestPow2(kRem);
        int64_t kShape = prefetchWidth;
        auto insertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPoint(prevDot);
        auto aEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getA().getType().getEncoding());
        auto bEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getB().getType().getEncoding());
        assert(aEncoding && bEncoding && "A and B must have DotOperandEncodingAttr types.");
        if (!dotIsSparse || (dotIsSparse && kOff % (2*prefetchWidth) == 0)) { // TODO : (Arya) Make this more pretty if you can
          aArg = generatePrefetch(mapping.lookup(dot2aLoopArg[dot]), 0,
                                  aEncoding.getMeta(), false, dotEncoding,
                                  builder, kOff, kShape);
          cloneElementwiseOps(aArg, dot2aVals[dot], builder);
        }
        Value bRem =
            generatePrefetch(mapping.lookup(dot2bLoopArg[dot]), 1, bEncoding.getMeta(),
                             false, dotEncoding, builder, kOff, kShape);
        cloneElementwiseOps(bRem, dot2bVals[dot], builder);

        if (dotIsSparse && (spSel == 0 || !canUseLdmatrix)) {
          auto eEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getE().getType().getEncoding());
          eArg = generatePrefetch(mapping.lookup(dot2eLoopArg[dot]), eEncoding.getOpIdx(),
                                  eEncoding.getMeta(), false, dotEncoding, builder,
                                  kOff, metadataPrefetchWidth);
          cloneElementwiseOps(eArg, dot2eVals[dot], builder);
        }

        builder.restoreInsertionPoint(insertionPoint);
        newOp = builder.clone(*dot, mapping);
        newOp->setOperand(0, aArg);
        newOp->setOperand(1, bRem);
        newOp->setOperand(2, prevDot->getResult(0));
        if (dotIsSparse) {
          newOp->setOperand(3, eArg);
          auto _dot = dyn_cast<triton::DotOp>(newOp);
          _dot.setSpSelector(spSel);
          spSel = (spSel + 1)%4;
        }

        prevDot = newOp;
        kOff += kShape;
        kRem -= kShape;
      }
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    Attribute dotEncoding = dot.getType().getEncoding();
    auto aEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getA().getType().getEncoding());
    auto bEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getB().getType().getEncoding());
    assert(aEncoding && bEncoding && "A and B must have DotOperandEncodingAttr types.");

    Value aToYield = generatePrefetch(mapping.lookup(dot2aYield[dot]), 0, aEncoding.getMeta(),
                                      true, dotEncoding, builder);
    cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
    yieldValues.push_back(aToYield);
    // bToYield
    Value bToYield = generatePrefetch(mapping.lookup(dot2bYield[dot]), 1, bEncoding.getMeta(),
                                      true, dotEncoding, builder);
    cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
    yieldValues.push_back(bToYield);

    if (triton::isSparseDot(dot)) {
      DotOperandEncodingAttr eEncoding = dyn_cast<DotOperandEncodingAttr>(dot.getE().getType().getEncoding());
      Value eToYield = generatePrefetch(mapping.lookup(dot2eYield[dot]), eEncoding.getOpIdx(),
                                        eEncoding.getMeta(), true, dotEncoding, builder);
      cloneElementwiseOps(eToYield, dot2eVals[dot], builder);
      yieldValues.push_back(eToYield);
    }
  }
  // Update ops of yield
  if (!yieldValues.empty())
    builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
  return newForOp;
}

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsAndFoldGreedily(getOperation(),
                                           std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
