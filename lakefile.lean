import Lake
open Lake DSL

package formal_verif_ml {
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git"

lean_lib FormalVerifML where
  srcDir := "lean"
  roots := #[`FormalVerifML]

@[default_target]
lean_exe formal_verif_ml_exe where
  root := `FormalVerifML.Main
  supportInterpreter := true
  srcDir := "lean"
