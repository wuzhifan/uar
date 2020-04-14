Initialization: Initialize the maxround, τij, ant_count, alpha, beta, rho, q1, q2.
for each times ∈ [1, maxround] do
    for each ant k ∈ [1, ant_count] do
        Set the ban set C of the ant k to be emptied.
        while C is not a full sink node set do
            Ant k selects the next position j according to
            the state transition probability by (27).
            Add location j to the ban set and move the ant to
            the new location.
        end while
        Update the pheromone table according to (29), (30)  and (31).
    end for
end for
End.