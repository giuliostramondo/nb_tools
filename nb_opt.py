import math

def computeReward(delegated, valStake, valOwned, rewardToDistribute, eta, pr_commission=False, commission=0.05 ):
    proportionalReward = rewardToDistribute * (1-eta)
    nakamotoBonus = rewardToDistribute - proportionalReward
    nakamotoBonusPerVal = nakamotoBonus / len(valStake)
    totalAtStake = sum(valStake) + sum(delegated)
    reward = 0
    reward_from_commissions = 0
    for v in range(len(valStake)):
        if delegated[v] > 0 or valOwned[v]:
            validatorShare =  (delegated[v] + valStake[v])/totalAtStake
            validator_total_reward = proportionalReward * validatorShare + nakamotoBonusPerVal
            if pr_commission:
                validator_commission = proportionalReward * validatorShare * commission
            else:
                validator_commission = validator_total_reward * commission
            validator_reward_to_distribute = validator_total_reward - validator_commission
            delegatorShare = delegated[v]/(delegated[v] + valStake[v])
            if valOwned[v]:
                reward += validator_reward_to_distribute * delegatorShare + validator_commission
                reward_from_commissions += validator_commission
            else:
                reward += validator_reward_to_distribute * delegatorShare
    return reward, reward_from_commissions


def findOptimalDelegation(amtDelegation, valStake, valOwned, minDelegation, rewardToDistribute, eta ,pr_commissions=False):

    R = rewardToDistribute
    v = list(valStake)
    o = list(valOwned)
    c = [0.05 for _ in valOwned]   # commissions per-validator c_i = 0.05
    d = amtDelegation
    TV = sum(valStake) + d
    m = list(minDelegation)
    N = len(v)

    tol = 1e-9
    # max_iter = 5000000  # allow more iterations for convergence
    #max_iter = 60000000  # allow more iterations for convergence
    max_iter = 200000  # allow more iterations for convergence

    assert len(c) == N and len(o) == N and len(m) == N, "v, c, o, m must have same length"

    # --- Feasibility check: must at least satisfy lower bounds ---
    m_sum = sum(m)
    if d < m_sum - tol:
        #raise ValueError(f"Validator cannot create a Sybil: d={d} < sum(m)={m_sum}")
        return 0, 0, [ 0 for i in m ]
    if abs(d - m_sum) <= tol:
        # exactly at bounds
        reward, _ = computeReward(m, valStake, valOwned, rewardToDistribute, eta ,pr_commissions)
        return reward, list(m)

    ValN = N
    A0 = R * (1.0 - eta) / TV       # R(1-eta)/TV
    B0 = R * eta / ValN             # R*eta/ValN

    # ----- helper: projection onto { x_i >= m_i, sum x_i = d } -----
    def project_onto_simplex_with_lower_bounds(z, m, d):
        """
        Project z onto { x : x_i >= m_i, sum x_i = d }.
        Do it via y = x - m, project y onto simplex { y_i >= 0, sum y_i = d - sum m_i }.
        """
        N = len(z)
        m_sum = sum(m)
        B = d - m_sum
        if B < 0:
            #raise ValueError(f"Validator cannot create a Sybil: d={d} < sum(m)={m_sum}")
            return 0, 0, [ 0 for i in m ]
        if abs(B) <= 1e-18:
            return list(m)

        # y_raw is tentative (before projection) excess above m
        y_raw = [z[i] - m[i] for i in range(N)]

        # Standard simplex projection (Duchi et al.)
        u = sorted(y_raw, reverse=True)
        cssv = [0.0] * N
        cssv[0] = u[0]
        for i in range(1, N):
            cssv[i] = cssv[i - 1] + u[i]

        rho = -1
        for i in range(N):
            t = (cssv[i] - B) / (i + 1)
            if u[i] - t > 0:
                rho = i
        if rho == -1:
            theta = 0.0
        else:
            theta = (cssv[rho] - B) / (rho + 1)

        y = [max(0.0, y_raw[i] - theta) for i in range(N)]
        x = [m[i] + y[i] for i in range(N)]
        return x

    # ----- gradient of your f wrt x_i -----
    # f_i(x_i) = ( R*(1-eta)*(x_i + v_i)/TV + R*eta/ValN ) * ( (1-c_i)*x_i/(x_i + v_i) + o_i*c_i )
    # df_i/dx_i = A0 * Term2 + Term1 * (1-c_i)*v_i/(x_i + v_i)^2
    def grad_F(x):
        g = []
        for i in range(N):
            xi = x[i]
            vi = v[i]
            ci = c[i]
            oi = o[i]

            denom = xi + vi
            # avoid degenerate case
            if denom <= 0:
                denom = 1e-18

            term1 = R * (1.0 - eta) * (xi + vi) / TV + R * eta / ValN
            term2 = (1.0 - ci) * xi / denom + oi * ci

            dterm2_dx = (1.0 - ci) * vi / (denom * denom)

            # derivative df_i/dx_i
            dfi = A0 * term2 + term1 * dterm2_dx
            g.append(dfi)
        return g

    # Start from lower bounds plus uniform leftover
    leftover = d - m_sum
    x = [m[i] + leftover / N for i in range(N)]

    # initial reward
    prev_reward,_ = computeReward(x, valStake, valOwned, rewardToDistribute, eta ,pr_commissions)

    # projected gradient ascent
    for it in range(max_iter):
        g = grad_F(x)

        # scale step size adaptively
        # base step ~ fraction of delegation / (1 + sqrt(it))
        grad_norm = math.sqrt(sum(gi * gi for gi in g)) + 1e-18
        base = 0.5  # can tune if needed
        step_size = base * d / grad_norm / math.sqrt(it + 1.0)

        # gradient ascent step
        z = [x[i] + step_size * g[i] for i in range(N)]

        # project back to feasible region
        x_new = project_onto_simplex_with_lower_bounds(z, m, d)

        # check convergence via change in x
        max_diff = max(abs(x_new[i] - x[i]) for i in range(N))
        x = x_new

        # optionally also check reward improvement
        reward, _ = computeReward(x, valStake, valOwned, rewardToDistribute, eta ,pr_commissions)
        if abs(reward - prev_reward) < tol * max(1.0, abs(prev_reward)) and max_diff < tol:
            prev_reward = reward
            break

        prev_reward = reward

    # final reward & allocation
    final_reward, reward_from_commissions = computeReward(x, valStake, valOwned, rewardToDistribute, eta ,pr_commissions)
    return final_reward, reward_from_commissions, x

def getValStakes(stakingState, redelegateFrom):
    # Returns a list of validator stakes to use for the optimizer
    # removes from the stakes the amount delegated from accounts in
    # redelegateFrom. 
    # Returns:
    # amount : total amount in the redelegateFrom accounts
    # valStake: list of validator stakes
    # valopers: list of valoper address matching the order in valStake
    # monikers: list of monikers matching order in valStake
    valStake = []
    valopers = []
    monikers = []
    amount = 0
    redelegationSet = set(redelegateFrom)
    for v in stakingState.validators:
        if v.status == 'BOND_STATUS_BONDED':
            delegationAmtToRemove = 0
            for d in v.delegations:
                if d[0] in redelegateFrom:
                    delegationAmtToRemove += d[1]
                    print(f"Removing {d[1]} delegation from {v.moniker} (addr: {d[0]})")
            staked = v.delegator_shares - delegationAmtToRemove
            amount += delegationAmtToRemove
            valStake += [staked]
            monikers += [v.moniker]
            valopers += [v.valoper]
    return amount, valStake, valopers, monikers




def optimize(stakingState, amount, ownedVals, redelegateFrom, nbCoefficient, pr_commissions=False):
    redelegatedAmt, valStake, valopers, monikers = getValStakes(stakingState, redelegateFrom)
    amount += redelegatedAmt
    minDelegation = [0 for v in valStake]
    rewardToDistribute = 100
    valOwned = []
    for i in range(len(valStake)):
        if valopers[i] in ownedVals:
            valOwned += [True]
        else:
            valOwned += [False]

    print("Amount to delegate: "+str(amount))
    # print("ValStake " + str(valStake))
    # print("Monikers " + str(monikers))
    # print("valOwned " + str(valOwned))
    # print("minDelegation " + str(minDelegation))
    print("rewardToDistribute " + str(rewardToDistribute))
    print("nbCoefficient " + str(nbCoefficient))
    reward,reward_from_commissions, x_opt = findOptimalDelegation(amount, valStake, valOwned, minDelegation, rewardToDistribute, nbCoefficient, pr_commissions)
    print(f"Optimal reward: {reward}")
    # print(f"x_opt: {x_opt} sum : {sum(x_opt)}")
    for i in range(len(valStake)):
        if x_opt[i] > 0:
            print(f"{monikers[i]}: {x_opt[i]} ({x_opt[i]/amount*100:.2f}%)")

def sybilSimulation(stakingState, amount, ownedVals, redelegateFrom, nbCoefficients, pr_commissions=False):
    redelegatedAmt, valStake, valopers, monikers = getValStakes(stakingState, redelegateFrom)

    amount += redelegatedAmt
    minDelegation = [0 for v in valStake]
    rewardToDistribute = 100
    valOwned = []
    for i in range(len(valStake)):
        if valopers[i] in ownedVals:
            valOwned += [True]
        else:
            valOwned += [False]

    print(f"eta, totalReward, rewardFromCommissions, rewardFromDelegations, totalRewardSybil, rewardFromCommissionsSybil, rewardFromDelegationsSybil, diff%")
    for nbCoefficient in nbCoefficients:
        reward, reward_from_commissions, x_opt = findOptimalDelegation(amount, valStake, valOwned, minDelegation, rewardToDistribute, nbCoefficient, pr_commissions)
        # Find validator with smallest stake not owned
        min = valStake[0]
        min_pos = 0
        for i in range(len(valStake)):
            if valStake[i] < min and not valOwned[i]:
                min_pos = i
                min = valStake[i]
        newValStake = []
        for i in range(len(valStake)):
            if i != min_pos:
                if not valOwned[i]:
                    newValStake += [ valStake[i] + (min / (len(valStake) - 2)) ]
                    # newValStake += [ valStake[i]  ]
                if valOwned[i]:
                    newValStake += [ valStake[i] ]
            else:
                    newValStake += [ 0 ]
                    minDelegation[i] = valStake[i] + 1
                    valOwned[i] = True
        rewardSybil, reward_from_commissionsSybil, x_optSybil = findOptimalDelegation(amount, newValStake, valOwned, minDelegation, rewardToDistribute, nbCoefficient , pr_commissions)
        print(f"{nbCoefficient}, {reward}, {reward_from_commissions}, {reward-reward_from_commissions}, {rewardSybil}, {reward_from_commissionsSybil},{rewardSybil-reward_from_commissionsSybil}, {( (rewardSybil/(reward+0.0001))-1) * 100:.2f}")
        valOwned[min_pos] = False
        minDelegation = [0 for v in valStake]

def test(amount, valStake, valOwned, nbCoefficients, sybil = False):
    rewardToDistribute = 100
    minDelegation = [0 for v in valStake]
    for nbCoefficient in nbCoefficients:
        # print("Amount to delegate: "+str(amount))
        # print("ValStake " + str(valStake))
        # print("valOwned " + str(valOwned))
        # print("minDelegation " + str(minDelegation))
        # print("rewardToDistribute " + str(rewardToDistribute))
        # print("nbCoefficient " + str(nbCoefficient))
        reward,reward_from_commissions, x_opt = findOptimalDelegation(amount, valStake, valOwned, minDelegation, rewardToDistribute, nbCoefficient)
        if sybil:
            # Find validator with smallest stake not owned
            min = valStake[0]
            min_pos = 0
            for i in range(len(valStake)):
                if valStake[i] < min and not valOwned[i]:
                    min_pos = i
                    min = valStake[i]
            newValStake = []
            for i in range(len(valStake)):
                if i != min_pos:
                    if not valOwned[i]:
                        newValStake += [ valStake[i] + (min / (len(valStake) - 2)) ]
                        # newValStake += [ valStake[i]  ]
                    if valOwned[i]:
                        newValStake += [ valStake[i] ]
                else:
                        newValStake += [ 0 ]
                        minDelegation[i] = valStake[i] + 1
                        valOwned[i] = True
            rewardSybil, reward_from_commissionsSybil, x_optSybil = findOptimalDelegation(amount, newValStake, valOwned, minDelegation, rewardToDistribute, nbCoefficient)
            print(f"eta: {nbCoefficient} reward: {reward} rewardSybil: {rewardSybil}, diff: {( (rewardSybil/(reward+0.0001))-1) * 100:.2f}% delegations: {x_opt}/{x_optSybil}")
            valOwned[min_pos] = False
        
                


    
