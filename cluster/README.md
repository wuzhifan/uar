Initialization: Initialize the maxround, n, D, ld, lb, lf, E0, Ef, Ti, pr, p0, w1, w2 and f.
(Find cluster number k of KMeans using elbew method)
for each k ∈ [1, n] do
    Calculat KMeans SSE slop S
    if S <= slop threshold then
        Use k as the cluster number
        Break
    end if
end for
Initialize Clusters and find center nearest node as CH.
for each round ∈ [1, maxround] do
    Calculate each alive CNs energy consumption by (15)
    Calculate head energy consumption by (13)
    Refresh head candidate list.
    for each node in head candidate list do
        Calculate head electe threshold T(s) by (25)
        if random number <= T(s) then
            Set node as cluster head.
            Remove node from head candidates.
        end if
    end for
end for
    

