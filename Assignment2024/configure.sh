source modules.sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.add("JSON")'
julia --project=. -e 'using Pkg; Pkg.add("MPI")'
julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using MPIPreferences;use_system_binary(;force=true)'
julia --project=.  -e 'using Pkg; Pkg.precompile()'
julia --project=. -O3 --check-bounds=no -e 'using Pkg; Pkg.precompile()'
