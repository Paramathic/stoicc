set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(LLVM_TARGET_DEFINITIONS TritonOps.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OpsEnums.cpp.inc -gen-enum-defs)
add_mlir_doc(TritonOps TritonOps dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TritonAttrDefs.td)
mlir_tablegen(TritonAttrDefs.h.inc -gen-attrdef-decls)
mlir_tablegen(TritonAttrDefs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(TritonAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS TritonDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_mlir_doc(TritonDialect TritonDialect dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TritonTypes.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs)

set(LLVM_TARGET_DEFINITIONS TritonInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)

set(LLVM_TARGET_DEFINITIONS TritonTypeInterfaces.td)
mlir_tablegen(TritonTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TritonTypeInterfaces.cpp.inc -gen-type-interface-defs)

add_public_tablegen_target(TritonTableGen)

