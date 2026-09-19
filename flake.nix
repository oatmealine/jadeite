{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/26.05";
    unsloth-nix.url = "github:Daaboulex/unsloth-nix";
    # !! DO NOT REMOVE !!, fixes regressions introduced in nixpkgs-unstable
    unsloth-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, unsloth-nix, ... }: let
    lib = nixpkgs.lib;
    forAllSystems = fun:
      nixpkgs.lib.genAttrs [
        "x86_64-linux"
        # untested, but could work
        #"aarch64-linux"
      ] (system: fun {
        inherit system;
        pkgs = import nixpkgs {
          inherit system;
        };
      });
  in {
    devShells = forAllSystems ({ pkgs, system, ... }: let
      unslothShells = unsloth-nix.devShells.${system};
      makeShell = env: let
        unslothShell = unslothShells.${if env == "rocm" then "rocm" else "default"};
      in pkgs.mkShell {
        packages =
          # lazy way of doing this. works tho
          unslothShell.nativeBuildInputs
          # some tooling expects `rocminfo` to be in $PATH
          ++ lib.optional (env == "rocm") pkgs.rocmPackages.rocminfo;
      };
    in rec {
      cuda = makeShell "cuda";
      rocm = makeShell "rocm";
      default = lib.derivations.warnOnInstantiate "assuming CUDA; specify #cuda or #rocm as the shell to avoid this" cuda;
    });
  };
}
