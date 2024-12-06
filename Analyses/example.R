# Example R script to run a very simple SIS*n model with no age groups or vaccination
# BJS December 2024

require(deSolve)
# Function to calculate the derivatives of the SIS*n model
SIS_deltas <- function(t, state, parms){
    N_S <- parms$N_S
    birth_rate <- parms$birth_rate
    WANE <- parms$WANE
    REC_UP <- parms$REC_UP
    REC_SAME <- parms$REC_SAME
    S_REL <- parms$S_REL
    BETA <- parms$BETA
    SEASONALITY <- parms$SEASONALITY
    OFFSET <- parms$OFFSET
    delta <- rep(0, length(state))
    pop_size <- sum(state)
    for(i in 1:N_S){
        # Susceptible compartment i, add births, minus deaths, minus infections, plus/minus waning, plus recovery
        delta[2*i] <- ifelse(i==1,1,0)*birth_rate*pop_size - birth_rate*state[2*i] - S_REL[i]*BETA*(sum(state[seq(3, length(state), by=2)])/pop_size)*state[2*i]*(1 + SEASONALITY*sin(2*pi*(t/365 + OFFSET))) - WANE[i]*state[2*i] + WANE[i+1]*state[2*(i+1)] + REC_UP[i]*state[2*i-1] + REC_SAME[i+1]*state[2*i+1]
        # Infected compartment i, minus deaths, plus infections, minus recovery
        delta[2*i+1] <- - birth_rate*state[2*i+1] + S_REL[i]*BETA*(sum(state[seq(3, length(state), by=2)])/pop_size)*state[2*i]*(1 + SEASONALITY*sin(2*pi*(t/365 + OFFSET))) - REC_UP[i+1]*state[2*i+1] - REC_SAME[i+1]*state[2*i+1]
    }
    return(list(delta, state))
}
# Define inital state:
# state[1] = 0 (dummy)
# state[2] = initial susceptibles in class 1
# state[3] = initial infecteds in class 1
# state[4] = initial susceptibles in class 2
# state[5] = initial infecteds in class 2
# state[6] = 0 (dummy)
state <- rep(0, 6)
state[2] <- 1000
state[3] <- 1
# solve based on initial state with ode solver
parms <- list(N_S=2, birth_rate=0.001, WANE=c(0, 0, 0.1), REC_UP=c(0, 0.05, 0), REC_SAME=c(0, 0, 0.05), S_REL=c(1, 0.5), BETA=0.101, SEASONALITY=0.1, OFFSET=0.1)
out <- ode(y=state, times=seq(0, 10000, by=1), func=SIS_deltas, parms=parms)
print(head(out))
# plot
matplot(out[,1], out[,4] + out[,6], type="l", xlab="Time", ylab="Infected")
# save plot
png("R_example.png")
dev.off()