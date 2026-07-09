## BJS June 2025
## JAX and NumPyro compatible ODES

import jax
import jax.numpy as jnp


interp_fn = jax.vmap(jnp.interp, in_axes=(None, None, 1), out_axes=0)

def deltas(t, state, args):
    (FULL_POINTS, AGING_RATE, BIRTH_RATE, CONTACT_MATRIX, # population parameters
    BETA, WANE, S_REL, I_REL, P_OBS, OBS_AGE, RELATIVE_CONTACT, VAX_RATE, MATERNAL_IMMUNITY, # fit parameters
    REC_UP, REC_SAME, IMPORT_STRENGTH, NAG) = args # pathogen parameters
    N_S = 3
    delta = jnp.zeros((2*N_S+1,NAG))
    maternal = state[0]
    delta_maternal = 0
    shaped_state = state[1:].reshape((2*N_S+1,NAG))
    age_pops = jnp.sum(shaped_state[:2*N_S, :], axis=0).at[0].add(maternal)
    pop_size = jnp.sum(age_pops)
    infectious = shaped_state[1:2*N_S:2, :]
    susceptible = shaped_state[0:2*N_S:2, :]
    # # births
    birth_rate = jnp.interp(t, FULL_POINTS, BIRTH_RATE)
    maternal_immunity = jnp.sum(interp_fn(t, FULL_POINTS, MATERNAL_IMMUNITY) * susceptible[:,NAG-3])  / jnp.sum(susceptible[:,NAG-3])
    delta_maternal = delta_maternal + maternal_immunity*birth_rate*pop_size
    delta = delta.at[0,0].add((1-maternal_immunity)*birth_rate*pop_size)
    # infections - calculate force of infection
    relative_contact = interp_fn(t, FULL_POINTS, RELATIVE_CONTACT)
    import_strength = jnp.interp(t, FULL_POINTS, IMPORT_STRENGTH)
    CONTACT_t = relative_contact[:, None] * relative_contact[None, :] * CONTACT_MATRIX
    infectious_by_age = jnp.sum(I_REL * infectious, axis=0)
    infectious_contact = jnp.dot(CONTACT_t, infectious_by_age/age_pops)
    import_contact = import_strength*jnp.sum(CONTACT_t, axis=1)
    force_of_infection = BETA*(infectious_contact + import_contact)
    # multiply by susceptibles in each age group and susceptibility class
    new_infections = S_REL[:, None] * force_of_infection[None, :] * susceptible
    # update deltas for infections
    delta = delta.at[0:2*N_S:2, :].add(-new_infections)
    delta = delta.at[1:2*N_S:2, :].add(new_infections)
    # waning
    waners = WANE[1:, None] * susceptible[1:, :]
    delta = delta.at[2:2*N_S:2, :].add(-waners)
    delta = delta.at[0:2*(N_S-1):2, :].add(waners)
    # recovery
    recers_up = REC_UP[:-1, None] * infectious[:-1, :]
    recers_same = REC_SAME[:, None] * infectious
    delta = delta.at[1:2*(N_S-1):2, :].add(-recers_up)
    delta = delta.at[2:2*N_S:2, :].add(recers_up)
    delta = delta.at[1:2*N_S:2, :].add(-recers_same)
    delta = delta.at[0:2*N_S:2, :].add(recers_same)
    # aging
    agers = AGING_RATE[None, :]*shaped_state[:2*N_S, :]
    delta = delta.at[:2*N_S, :].add(-agers)
    delta = delta.at[:2*N_S, 1:].add(agers[:,:-1])
    # maternal compartment aging
    if NAG == 65:
        maternal_agers = (1 / ((3 / 12) * 365)) * maternal
        delta_maternal = delta_maternal - maternal_agers
        delta = delta.at[0, 3].add(maternal_agers)
    else:
        maternal_agers = AGING_RATE[0] * maternal
        delta_maternal = delta_maternal - maternal_agers
        delta = delta.at[0, 1].add(maternal_agers)
    # vaccination
    vrate = interp_fn(t, FULL_POINTS, VAX_RATE)
    # assume that S_VAX is the last susceptibility class
    vaxxers = vrate[None, :] * susceptible[:-1, :]
    delta = delta.at[0:2*(N_S-1):2, :].add(-vaxxers)
    delta = delta.at[2*N_S-2, :].add(jnp.sum(vaxxers, axis=0))
    # observations
    observed_new_incidence = P_OBS[:, None] * OBS_AGE[None,:] * new_infections
    delta = delta.at[-1, :].add(jnp.sum(observed_new_incidence, axis=0))
    # concatenate maternal immunity delta to flattened delta
    overall_delta = jnp.concatenate((jnp.array([delta_maternal]), delta.flatten()))
    return overall_delta