using JSON

N_values = [600, 1200, 2400, 4000]
P_values = [2, 4, 5, 8]

# Loop over N = 600 1200 2400 4000 and print speed-up and efficiency for P=2, 4, 5, 8

for N in N_values
    for P in P_values
        seq_data = JSON.parsefile("results/$(N)_seq.json")
        par_data = JSON.parsefile("results/$(N)_$(P).json")

        seq_time = seq_data["min_timings"]
        par_time = par_data["min_timings"]

        speedup = seq_time / par_time
        efficiency = speedup / P

        println("N = $N, P = $P, Speedup = $speedup, Efficiency = $efficiency")

    end
    println()
end