import Lake
open Lake DSL

package dl_demos_derivations

@[default_target]
lean_lib chapters where
  srcDir := "."
  roots := #[`chapters]
