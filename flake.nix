{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/26.05";
    unsloth-nix.url = "github:Daaboulex/unsloth-nix";
    # !! DO NOT REMOVE !!, fixes regressions introduced in nixpkgs-unstable
    unsloth-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, unsloth-nix, ... }: let
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
    packages = forAllSystems ({ pkgs, ... }: let
      python = pkgs.python3;
      pypkgs = python.pkgs;
    in {
      py-cord = pypkgs.buildPythonPackage rec {
        pname = "py-cord";
        version = "2.8.1";
        src = pkgs.fetchFromGitHub {
          owner = "Pycord-Development";
          repo = "pycord";
          tag = "v${version}";
          sha256 = "sha256-HsVzfQPRPZ09UZeK25+obzzBTbbAXcVujeAkYpxN11c=";
        };
        format = "pyproject";
        # maybe needs libffi here somehow?
        nativeBuildInputs = with pypkgs; [ setuptools setuptools-scm ];
        propagatedBuildInputs = with pypkgs; [
          aiohttp
          # speed
          msgspec
          #aiohttp[speedups] # the default on nixpkgs
          # voice
          #pynacl
          #davey
        ];

        pythonImportsCheck = [
          "discord"
          "discord.types"
          "discord.ui"
          "discord.webhook"
          "discord.commands"
          "discord.ext.commands"
          "discord.ext.tasks"
        ];

        patchPhase = ''
          #substituteInPlace "discord/opus.py" \
          #  --replace-fail 'ctypes.util.find_library("opus")' '"${pkgs.libopus}/lib/libopus${pkgs.stdenv.hostPlatform.extensions.sharedLibrary}"'

          #substituteInPlace "discord/player.py" \
          #  --replace-fail 'executable: str = "ffmpeg"' 'executable: str = "${pkgs.lib.getExe pkgs.ffmpeg}"'

          # relax setuptools and setuptools-scm dep version
          sed -i 's/,<=80.9.0//g' pyproject.toml
          sed -i 's/>=9.2,<=9.2.2//g' pyproject.toml
        '';
      };
    });
    devShells = forAllSystems ({ pkgs, system, ... }: let
      unslothShells = unsloth-nix.devShells.${system};
      makeShell = env: let
        unslothShell = unslothShells.${if env == "rocm" then "rocm" else "default"};
      in pkgs.mkShell {
        packages =
          # lazy way of doing this. works tho
          unslothShell.nativeBuildInputs
          # some tooling expects `rocminfo` to be in $PATH
          ++ lib.optional (env == "rocm") pkgs.rocmPackages.rocminfo
          ++ [
            pkgs.python3Packages.faker
            pkgs.python3Packages.atproto
          ];
      };
    in rec {
      cuda = makeShell "cuda";
      rocm = makeShell "rocm";
      default = lib.derivations.warnOnInstantiate "assuming CUDA; specify #cuda or #rocm as the shell to avoid this" cuda;
    });
  };
}
