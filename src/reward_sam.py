class reward_sam:
    def __init__(self, actor):
        self.actor = actor

    def __call__(self, errors, linear_vel, angular_vel, params):
        # get the matrix of all linguistic variables values' membership
        fuzz = self.actor.fuzzified

        # get the gains used in reward function calculation
        dis_gain            = params['dis_gain']
        theta_near_gain     = params['theta_near_gain']
        theta_recovery_gain = params['theta_recovery_gain']

        # decompose errors into individual variables with theta_recovery being the same thing as theta_far
        if errors.shape[0] == 5:
            target, dis, theta_lookahead, theta_recovery, theta_near = errors
        else:
            dis, theta_recovery, theta_near = errors
            target = 0
            theta_lookahead = 0

        # calculations related to a given variable as used in the reward functions below
        dis            = 1 / (           abs(dis) + dis_gain)
        theta_near     = 1 / (    abs(theta_near) + theta_near_gain)
        theta_recovery = 1 / (abs(theta_recovery) + theta_recovery_gain)

        # get the levels of membership in distLine for farLight and farRight (since those rules employ theta_recovery)
        farLeft  = fuzz[0, 1, 0].item()
        farRight = fuzz[0, 1, 6].item()
        theta_rcvy_memb = max(farLeft, farRight)

        # combine the reward functions in proportion to the membership in theta_recovery
        r1 = -dis * theta_near
        r2 = -dis * theta_recovery
        rewards = r1 * (1 - theta_rcvy_memb) + r2 * theta_rcvy_memb

        return rewards, [dis, theta_near, theta_recovery, linear_vel, angular_vel, theta_lookahead]
