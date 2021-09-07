"""
Implementation of the streaming peicewise linear regression and hypothesis testing
    framework for partitioning and reconstructing simulation outputs.

Paper: Myers et al (2016). Partitioning a Large Simulation as It Runs. 

Author: Wayne Wang
Email: wayneyw@umich.edu
Last modified: 08/02/2021
"""


function stream_lm(; 
    p1::Int = 2, p2::Int = 4, buff_size::Int = 5, α::Real = 0.025, δ::Real = 0.001,
    sims_tot::Int = 200)
    """
    Streaming piecewise linear regression with modifed F test for change-point detection
    """
    # initialization
    curr_sims_tot = 0
    partitions, curr, buff, N_curr, lm_fit_buff, suff_stats_buff, lm_fit_curr, suff_stats_curr,
        lm_fit_curr_and_buff, suff_stats_curr_and_buff =
            init(buff_size)

    # compute linear partitions while simulation is running
    while curr_sims_tot <= sims_tot - buff_size - 1
        print(string(curr_sims_tot, "\n"))
        lm_fit_buff, suff_stats_buff, lm_fit_curr, suff_stats_curr,
        lm_fit_curr_and_buff, suff_stats_curr_and_buff = 
            update_partitions!(partitions, curr, buff,
                suff_stats_curr, lm_fit_curr,
                suff_stats_buff, lm_fit_buff,
                suff_stats_curr_and_buff, lm_fit_curr_and_buff,
                N_curr, buff_size, p1, p2, α, δ)
    end

    return partitions
end


function init(buff_size::Int)
    """
    Compute initial statistics
    """
    partitions = [] #each element a tuple: (start/end time stamps, sufficient statistics: RSS, beta0, beta1)
    curr_and_buff = simulation_loader(2 * buff_size)
    curr = curr_and_buff[1:buff_size]
    buff = curr_and_buff[buff_size + 1:end]
    lm_fit_curr, suff_stats_curr = update_sufficient_stats(curr)
    curr = [curr[1][2], curr[end][2]] #discard simulation outputs, save only start/end time stamps
    N_curr = buff_size
    lm_fit_buff, suff_stats_buff = update_sufficient_stats(buff)
    lm_fit_curr_and_buff, suff_stats_curr_and_buff = update_sufficient_stats(curr_and_buff) #{curr \cup buff}

    return partitions, curr, buff, N_curr, lm_fit_buff, suff_stats_buff, lm_fit_curr, suff_stats_curr,
        lm_fit_curr_and_buff, suff_stats_curr_and_buff 
end


function simulation_loader(N::Int; 
    ys::AbstractVector = y, ts::AbstractVector = x)
    """
    Genrate N new simulation outputs
    return a time-series tuples {(y1,t1),…,(yN,tN)}
    """
    sims = []
    for i = curr_sims_tot + 1:curr_sims_tot + N
        push!(sims, (ys[i], ts[i]))
    end
    global curr_sims_tot += N
    return sims
end


function update_partitions!(partitions::AbstractVector, 
    curr::AbstractVector, 
    buff::AbstractVector, 
    suff_stats_curr::AbstractVector,
    lm_fit_curr::AbstractVector,
    suff_stats_buff::AbstractVector,
    lm_fit_buff::AbstractVector,
    suff_stats_curr_and_buff::AbstractVector,
    lm_fit_curr_and_buff::AbstractVector,
    N_curr::Int, buff_size::Int, p1::Int, p2::Int, α::Real, δ::Real)
    """
    Update partitions, curr, and buff data/statistics/fit based on a modified F-test
    """
    # modified F-test
    N = N_curr + buff_size
    RSS_curr = lm_fit_curr[1]
    RSS_buff = lm_fit_buff[1]
    RSS1 = lm_fit_curr_and_buff[1]
    RSS2 = RSS_curr + RSS_buff
    F_stat = ((RSS1 - RSS2) / (p2 - p1)) / ((RSS2 + δ^2 * N) / (N - p2))
    F_crit = quantile(FDist(p2 - p1, N - p2), 1 - α)

    if F_stat > F_crit # reject null
        ## save partition info: start/end time stamps & regressioin fit  
        push!(partitions, (copy(curr), copy(lm_fit_curr)))
        curr .= [buff[1][2], buff[end][2]]
        N_curr = buff_size
        buff .= simulation_loader(buff_size)
        ## compute suff. stats
        lm_fit_curr .= copy(lm_fit_buff)
        suff_stats_curr .= copy(suff_stats_buff)
        lm_fit_buff, suff_stats_buff = update_sufficient_stats(buff)
        ## update lm_fit_curr_and_buff for each new point in buff
        lm_fit_curr_and_buff, suff_stats_curr_and_buff = 
                update_sufficient_stats(lm_fit_curr, suff_stats_curr, N_curr + buff_size, buff)
    else # fail to reject null
        curr[2] = buff[1][2]
        lm_fit_curr, suff_stats_curr = update_sufficient_stats(lm_fit_curr, suff_stats_curr, N_curr, [buff[1]])
        deleteat!(buff, 1)
        push!(buff, simulation_loader(1)[1])
        lm_fit_buff, suff_stats_buff = update_sufficient_stats(buff)
        lm_fit_curr_and_buff, suff_stats_curr_and_buff = 
                update_sufficient_stats(lm_fit_curr_and_buff, suff_stats_curr_and_buff, N_curr + buff_size, [buff[end]])
    end

    return lm_fit_buff, suff_stats_buff, lm_fit_curr, suff_stats_curr,
        lm_fit_curr_and_buff, suff_stats_curr_and_buff
end


function update_sufficient_stats(sims::AbstractVector)
    """
    Compute sufficient statistics & fitted regression line for vectors
        of simulation outputs sims := {(y1,t1),…}
    """
    suff_stats = zeros(5) #sumt, sumt2, sumy, sumy2, summty
    lm_fit = zeros(3) #RSS, intercept, slope
    N = 0

    lm_fit, suff_stats = update_sufficient_stats(lm_fit, suff_stats, N, sims)

    return lm_fit, suff_stats
end


function update_sufficient_stats(lm_fit::AbstractVector, suff_stats::AbstractVector,
    N::Int, new_sims::AbstractVector)
    """
    Update sufficient statistics & fitted regression line for one or more newly   
        added simulation outputs
    """ 
    # create copies to store the new stats
    lm_fit_new = similar(lm_fit) #RSS, intercept, slope
    suff_stats_new = copy(suff_stats) #sumt, sumt2, sumy, sumy2, summty

    for i = 1:length(new_sims)
        newy = new_sims[i][1]
        newt = new_sims[i][2]
        # update sufficient statistics
        suff_stats_new[1] += newt # sum of the time steps
        suff_stats_new[2] += newt * newt # sum of time steps squared
        suff_stats_new[3] += newy # sum of variable of interest (e.g. temp)
        suff_stats_new[4] += newy * newy # sum of variable of interest (e.g. temp)
        suff_stats_new[5] += newt * newy # sum of time cross variable
        N += 1
    end

    lm_fit_new .= compute_lm(N, suff_stats_new)

    return lm_fit_new, suff_stats_new
end


function compute_lm(N::Int, suff_stats::AbstractVector)
    """
    Compute linear regression fit (RSS, slope, intercept)
        using sufficient statistics
    """
    sumt = suff_stats[1]
    sumt2 = suff_stats[2]
    sumy = suff_stats[3]
    sumy2 = suff_stats[4]
    sumty = suff_stats[5]
    RSS = (sumty - sumt * sumy / N)^2
    RSS = RSS / (sumt2 - sumt * sumt / N)
    RSS = sumy2 - sumy * sumy / N - RSS
    #sigma2 = RSS / (N - 2)
    beta1 = sumty - sumt * sumy / N
    beta1 = beta1 / (sumt2 - sumt * sumt / N)
    beta0 = (sumy - beta1 * sumt) / N
    return [RSS, beta0, beta1]
end
