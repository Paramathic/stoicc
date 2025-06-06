#ifndef TRITON_PATTERNS
#define TRITON_PATTERNS

include "mlir/Dialect/Arith/IR/ArithOps.td"
include "triton/Dialect/Triton/IR/TritonOps.td"
include "mlir/IR/PatternBase.td"

// addptr(addptr(%ptr, %idx0), %idx1) => addptr(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithDialect
//         (ref: ArithCanonicalization.td)
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;
def CombineAddPtrPattern : Pat<
        (TT_AddPtrOp (TT_AddPtrOp $ptr, $idx0), $idx1),
        (TT_AddPtrOp $ptr, (Arith_AddIOp $idx0, $idx1, DefOverflow)),
        [(Constraint<CPred<"isAddPtrOffsetCombinable($0, $1)">> $idx0, $idx1)]>;

// broadcast(cst) => cst
def getConstantValue : NativeCodeCall<"getConstantValue($_builder, $0, $1)">;
def CombineBroadcastConstantPattern : Pat<
    (TT_BroadcastOp:$bcast_res (Arith_ConstantOp $value)),
    (Arith_ConstantOp (getConstantValue $value, $bcast_res), (location $bcast_res)),
    [(Constraint<CPred<"isBroadcastConstantCombinable($0)">> $value)]>;

#endif
